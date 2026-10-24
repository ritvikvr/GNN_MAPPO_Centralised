"""Discrete-time multi-UAV dynamic task-allocation simulator.

API (Gymnasium-style, multi-agent):
    env = UAVTaskEnv(cfg)
    obs, info = env.reset(seed=..., options={"controller": "decentralized" | "centralized"})
    obs, reward, terminated, truncated, info = env.step({uav_id: Action, ...})
``obs`` is a light dict (time + the UAVs that should be queried). Per-UAV *local* graph observations are built on
demand with ``env.observe(uav_ids)`` and the critic-only privileged state with ``env.global_state()``.

Time order inside one step ``t -> t+1``:
  1. apply actions (feasibility re-checked; requests sent over the lossy network)
  2. heartbeats are broadcast
  3. physics: motion, energy, service, charging, failures, wind
  4. task expiry, t += 1
  5. begin step: task arrivals + announcements, outages, message delivery, protocol tick (confirm / lease expiry),
     forced safety returns, ground-truth task status refresh.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..coordination.bidding import compute_bid
from ..coordination.protocol import CoordinationLayer
from ..safety.feasibility import FeasibilityChecker, TaskArrays
from .communication import CommNetwork
from .dynamics import WindModel, move_towards, nominal_travel_energy, service_energy
from .task import Task, TaskInfo, TaskStatus
from .uav import Action, ActionKind, UAV, UAVStatus

MAX_GLOBAL_UAV = 64
MAX_GLOBAL_TASK = 128
AT_BASE_RADIUS = 5.0


class UAVTaskEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg.validate()
        e = cfg.env
        if e.n_uavs > MAX_GLOBAL_UAV or e.n_tasks > MAX_GLOBAL_TASK:
            raise ValueError(f"n_uavs<={MAX_GLOBAL_UAV} and n_tasks<={MAX_GLOBAL_TASK} supported by the global critic encoding")
        pw = np.asarray(e.priority_weights, dtype=float)
        self.mean_value = float((pw / pw.sum()) @ np.arange(1, 6))
        self.t = 0
        self.uavs: List[UAV] = []
        self.tasks: List[Task] = []
        self.mode = "decentralized"

    # ================================================================== reset
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        cfg, e = self.cfg, self.cfg.env
        options = options or {}
        self.mode = options.get("controller", "decentralized")
        self.seed = int(seed if seed is not None else 0)
        ss = np.random.SeedSequence(self.seed)
        r_scn, r_dyn, r_com = [np.random.default_rng(s) for s in ss.spawn(3)]
        self.rng_dyn = r_dyn
        self.t = 0
        self.bases = self._make_bases()
        self.checker = FeasibilityChecker(e, cfg.safety, self.bases)
        self.uavs = self._make_uavs(r_scn)
        self.tasks = self._make_tasks(r_scn)
        self.wind = WindModel(e, r_dyn)
        self.comm = CommNetwork(cfg.comm, e.n_uavs, r_com)
        if r_com.random() < cfg.comm.partition_prob:
            start = int(r_com.integers(int(0.2 * e.horizon), max(int(0.6 * e.horizon), int(0.2 * e.horizon) + 1)))
            self.comm.schedule_partition(start, cfg.comm.partition_duration, e.arena / 2)
        self.coord = CoordinationLayer(e.n_uavs, cfg.coordination, cfg.comm, self.comm)
        self.last_decision = np.full(e.n_uavs, -10_000, dtype=np.int64)
        self.m: Dict[str, float] = defaultdict(float)
        self.hist: Dict[str, List[float]] = {"completed_value": [0.0], "energy": [0.0], "reward": [], "completed": [0.0]}
        self.comp_totals: Dict[str, float] = defaultdict(float)
        self.last_completion_t = 0
        self.ep_return = 0.0
        self.last_conflict_t = -1
        self._ever_committed: set = set()
        self._begin_step(defaultdict(float))
        return self._obs(), {"seed": self.seed}

    def _make_bases(self) -> np.ndarray:
        e = self.cfg.env
        if e.n_bases == 1:
            return np.array([[0.5 * e.arena, 0.5 * e.arena]])
        xs = e.arena * (0.1 + 0.8 * np.arange(e.n_bases) / (e.n_bases - 1))
        ys = e.arena * (0.5 + 0.25 * np.where(np.arange(e.n_bases) % 2 == 0, -1.0, 1.0) * (e.n_bases > 2))
        return np.stack([xs, ys], axis=1)

    def _make_uavs(self, rng: np.random.Generator) -> List[UAV]:
        e = self.cfg.env
        out: List[UAV] = []
        for i in range(e.n_uavs):
            b = self.bases[int(rng.integers(len(self.bases)))]
            pos = np.clip(b + rng.uniform(-40, 40, size=2), 0, e.arena)
            caps = rng.random(e.n_capabilities) < 0.5
            caps[i % e.n_capabilities] = True
            cap_e = e.capacity * (1.0 + e.capacity_hetero * rng.uniform(-1, 1))
            out.append(UAV(id=i, pos=pos, speed=e.speed * (1.0 + e.speed_hetero * rng.uniform(-1, 1)), capacity=cap_e,
                           energy=cap_e * rng.uniform(0.85, 1.0), payload_capacity=e.payload_capacity,
                           capabilities=caps, payload_rem=e.payload_capacity, base_id=-1))
        return out

    def _make_tasks(self, rng: np.random.Generator) -> List[Task]:
        e, s = self.cfg.env, self.cfg.safety
        n = e.n_tasks
        n0 = int(round(e.initial_task_frac * n))
        window = max(int(e.arrival_window_frac * e.horizon), 1)
        arr = np.concatenate([np.zeros(n0), np.sort(rng.integers(1, window + 1, size=n - n0))]).astype(int)
        pw = np.asarray(e.priority_weights, dtype=float)
        tasks: List[Task] = []
        zones = np.asarray(e.no_fly_zones, dtype=float).reshape(-1, 3)
        for k in range(n):
            for _ in range(30):
                pos = rng.uniform(0, e.arena, size=2)
                if not len(zones) or np.all(np.linalg.norm(zones[:, :2] - pos, axis=1) > zones[:, 2]):
                    break
            svc = float(rng.uniform(*e.service_range))
            slack = float(rng.uniform(*e.deadline_slack_range)) * e.deadline_tightness
            pay = float(rng.uniform(*e.task_payload_range)) if rng.random() < 0.6 else 0.0
            t = Task(id=k, pos=pos, required_capability=int(rng.integers(e.n_capabilities)),
                     priority=int(rng.choice(5, p=pw / pw.sum())) + 1, arrival_time=float(arr[k]),
                     deadline=float(arr[k]) + svc + slack, service_duration=svc, payload=pay)
            t.feasible_ever = self._feasible_ever(t)
            tasks.append(t)
        return tasks

    def _feasible_ever(self, t: Task) -> bool:
        """Optimistic achievability: some UAV (from a *full* battery, starting at its nearest base, ignoring queueing)
        could serve the task before its deadline. Used for the 'feasible-task completion rate' metric."""
        e, s = self.cfg.env, self.cfg.safety
        d = float(np.linalg.norm(self.bases - t.pos[None, :], axis=1).min())
        for u in self.uavs:
            if not u.capabilities[t.required_capability] or t.payload > u.payload_capacity:
                continue
            need = 2 * nominal_travel_energy(d, e) * (1 + s.energy_margin) + service_energy(t.service_duration, e)
            if need > u.capacity * (1 - s.reserve_frac):
                continue
            if t.arrival_time + d / u.speed * (1 + s.time_margin) + t.service_duration <= t.deadline or not e.hard_deadlines:
                return True
        return False

    # ================================================================== observation API
    def _obs(self) -> dict:
        return {"t": self.t, "decision_uavs": self.decision_uavs()}

    def decision_uavs(self) -> List[int]:
        """UAVs whose policy should be queried now. Safety-forced and unavailable UAVs are excluded."""
        out = []
        P = self.cfg.env.redecide_period
        for u in self.uavs:
            if u.status == UAVStatus.IDLE:
                if not self.checker.must_return(u):
                    out.append(u.id)
            elif u.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE) and self.t - self.last_decision[u.id] >= P:
                out.append(u.id)
        return out

    def observe(self, uav_ids: List[int]):
        from ..graph.graph_builder import GraphBuilder

        if not hasattr(self, "_builder") or self._builder.env is not self:
            self._builder = GraphBuilder(self)
        return [self._builder.build(i) for i in uav_ids]

    def global_state(self):
        from ..graph.observations import build_global_state

        return build_global_state(self)

    # ================================================================== step
    def step(self, actions: Dict[int, Action], centralized: Optional[bool] = None):
        if centralized is None:
            centralized = self.mode == "centralized"
        t = self.t
        comp: Dict[str, float] = defaultdict(float)
        self.comm.step_bytes = 0.0
        msgs0 = self.comm.stats["messages"]
        for uid in sorted(actions):
            self.last_decision[uid] = t
            if centralized:
                self._apply_command(uid, actions[uid], t, comp)
            else:
                self._apply_action(uid, actions[uid], t, comp)
        if not centralized and t % self.cfg.comm.heartbeat_period == 0:
            self.coord.send_heartbeats(t, self.uavs)
        self._physics(t, comp)
        self.t += 1
        self._expire_tasks(comp)
        self._begin_step(comp)
        reward = self._reward(comp)
        terminated = self._mission_over()
        truncated = (not terminated) and self.t >= self.cfg.env.horizon
        self.ep_return += reward
        self.hist["reward"].append(reward)
        self.hist["completed_value"].append(self.m["completed_value"])
        self.hist["completed"].append(self.m["completed"])
        self.hist["energy"].append(self.m["energy"])
        for k, v in comp.items():
            self.comp_totals[k] += v
        self.m["step_messages"] += self.comm.stats["messages"] - msgs0
        info: Dict[str, Any] = {"reward_components": dict(comp), "t": self.t}
        if terminated or truncated:
            info["episode"] = self.summary()
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ actions
    def _tasks_arrays(self, infos) -> TaskArrays:
        return TaskArrays.from_infos(infos)

    def _override(self, u: UAV, t: int, comp) -> None:
        """Safety override: replace an invalid/unsafe request by the safe fallback."""
        self.m["safety_overrides"] += 1
        self._release_commitment(u, t, comp, switch=False)
        if self.checker.must_return(u):
            self._do_return_or_emergency(u, t)
        else:
            if u.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE):
                u.status = UAVStatus.IDLE

    def _apply_action(self, uid: int, a: Action, t: int, comp) -> None:
        u = self.uavs[uid]
        if u.status == UAVStatus.FAILED:
            return
        tab = self.coord.tables[uid]
        if a.kind == ActionKind.CONTINUE:
            return
        if a.kind == ActionKind.WAIT:
            if u.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE):
                self._release_commitment(u, t, comp, switch=True)
                u.status = UAVStatus.IDLE
            return
        if a.kind == ActionKind.RETURN:
            if u.status in (UAVStatus.EXECUTING, UAVStatus.CHARGING, UAVStatus.RETURNING):
                return
            self._release_commitment(u, t, comp, switch=u.has_commitment)
            self._do_return_or_emergency(u, t)
            return
        # ---- TASK
        info = tab.tasks.get(a.task_id)
        if info is None or a.task_id in tab.done or u.status in (UAVStatus.EXECUTING, UAVStatus.RETURNING, UAVStatus.CHARGING):
            self.m["invalid_actions"] += 1
            return
        if u.task_id == a.task_id and tab.own is not None:
            return
        feas = self.checker.check(u, self._tasks_arrays([info]), t)
        if not feas.mask[0]:
            self._override(u, t, comp)
            return
        if self.cfg.coordination.enabled:
            ex = tab.claims.get(a.task_id)
            if ex is not None and ex.uav_id != uid and int(ex.status) >= 1 and t <= ex.expiry:
                self.m["invalid_actions"] += 1          # locally known CONFIRMED/EXECUTING claim by someone else
                return
        if u.has_commitment:
            self._release_commitment(u, t, comp, switch=True)
        task = self.tasks[a.task_id]
        slack = info.deadline - (t + float(feas.eta[0]) + info.service_duration)
        bid = compute_bid(info.priority, float(feas.eta[0]), info.service_duration, slack, float(feas.e_travel[0]), u.capacity)
        self.coord.request(uid, info, bid, t)
        u.task_id = a.task_id
        u.progress = 0.0
        u.status = UAVStatus.RESERVED if self.cfg.coordination.enabled else UAVStatus.EN_ROUTE
        self.m["requests"] += 1
        if not self.cfg.coordination.enabled:
            self._note_commit(task, comp)

    def _note_commit(self, task: Task, comp) -> None:
        """Book-keeping when a UAV becomes truly committed (en route) to a task."""
        self.m["commitments"] += 1
        if task.id in self._ever_committed:
            self.m["reassignments"] += 1
        self._ever_committed.add(task.id)
        others = [x for x in self.uavs if x.task_id == task.id and x.status in (UAVStatus.EN_ROUTE, UAVStatus.EXECUTING)]
        if len(others) > 1:
            self.m["concurrent_commitments"] += 1

    def _apply_command(self, uid: int, a: Action, t: int, comp) -> None:
        """Centralized controller -> UAV command over the (lossy) ground link."""
        u = self.uavs[uid]
        if u.status == UAVStatus.FAILED:
            return
        self.comm.stats["messages"] += 1
        self.comm.stats["bytes"] += 32
        self.comm.step_bytes += 32
        if (not u.comm_ok(t)) or self.comm.rng.random() < self.cfg.comm.command_loss:
            self.m["commands_lost"] += 1
            return
        if a.kind == ActionKind.TASK:
            if u.status not in (UAVStatus.IDLE,):
                return
            task = self.tasks[a.task_id]
            if task.status.terminal or task.status != TaskStatus.PENDING:
                self.m["invalid_actions"] += 1
                return
            info = TaskInfo.from_task(task)
            if not self.checker.check(u, self._tasks_arrays([info]), t).mask[0]:
                self.m["safety_overrides"] += 1
                return
            u.task_id, u.status, u.progress = task.id, UAVStatus.EN_ROUTE, 0.0
            task.status, task.owner = TaskStatus.ASSIGNED, uid
            self._note_commit(task, comp)
        elif a.kind == ActionKind.RETURN:
            if u.status in (UAVStatus.IDLE, UAVStatus.RESERVED, UAVStatus.EN_ROUTE):
                self._release_commitment(u, t, comp, switch=False)
                self._do_return_or_emergency(u, t)

    def _release_commitment(self, u: UAV, t: int, comp, switch: bool, completed: bool = False) -> None:
        if u.task_id >= 0:
            task = self.tasks[u.task_id]
            if not completed and not task.status.terminal:
                task.version += 1
        if self.mode != "centralized":
            self.coord.release(u.id, t, completed)
        if switch and u.has_commitment:
            comp["switch"] += 1
            u.switches += 1
            self.m["task_switches"] += 1
        u.task_id = -1
        u.progress = 0.0

    def _do_return_or_emergency(self, u: UAV, t: int) -> None:
        j, d = self.checker.nearest_base(u.pos)
        if not self.checker.can_return(u) and self.cfg.safety.emergency_policy == "land_in_place" and self.cfg.safety.energy_filter:
            self._fail(u, "emergency_landing")
            return
        u.base_id = j
        u.status = UAVStatus.CHARGING if d <= AT_BASE_RADIUS else UAVStatus.RETURNING
        if u.status == UAVStatus.CHARGING:
            u.pos = self.bases[j].copy()
        u.task_id = -1

    def _fail(self, u: UAV, reason: str) -> None:
        if u.status == UAVStatus.FAILED:
            return
        u.status, u.fail_reason, u.vel = UAVStatus.FAILED, reason, np.zeros(2)
        if reason == "emergency_landing":
            self.m["emergency_landings"] += 1
        elif reason == "energy_depleted":
            self.m["energy_depleted"] += 1
        else:
            self.m["uav_failures"] += 1
        u.task_id = -1   # table keeps (and stops renewing) its claim -> remote leases expire

    # ------------------------------------------------------------------ physics
    def _physics(self, t: int, comp) -> None:
        e, dt = self.cfg.env, self.cfg.env.dt
        wind = self.wind.step(dt)
        for u in self.uavs:
            if u.status == UAVStatus.FAILED:
                continue
            used = 0.0
            st = u.status
            u.vel = np.zeros(2)
            if st in (UAVStatus.IDLE, UAVStatus.RESERVED):
                used = e.hover_energy_per_s * dt
            elif st == UAVStatus.EN_ROUTE:
                task = self.tasks[u.task_id]
                if task.status.terminal or (task.status == TaskStatus.IN_PROGRESS and task.owner != u.id):
                    self._abort_duplicate(u, t, comp, task)
                    used = e.hover_energy_per_s * dt
                else:
                    u.pos, step, used, u.vel = move_towards(u.pos, task.pos, u.speed, wind, dt, e)
                    u.distance += step
                    if float(np.linalg.norm(task.pos - u.pos)) < 1e-6:
                        self._arrive(u, task, t, comp)
            elif st == UAVStatus.EXECUTING:
                task = self.tasks[u.task_id]
                used = e.service_energy_per_s * dt
                u.progress += dt / task.service_duration
                if u.progress >= 1.0 - 1e-9:
                    self._complete(u, task, t, comp)
            elif st == UAVStatus.RETURNING:
                u.pos, step, used, u.vel = move_towards(u.pos, self.bases[u.base_id], u.speed, wind, dt, e)
                u.distance += step
                if float(np.linalg.norm(self.bases[u.base_id] - u.pos)) < 1e-6:
                    u.status = UAVStatus.CHARGING
            elif st == UAVStatus.CHARGING:
                u.energy = min(u.capacity, u.energy + e.charge_rate * dt)
                if u.energy >= 0.95 * u.capacity:
                    u.payload_rem = u.payload_capacity
                    u.status = UAVStatus.IDLE
            u.energy -= used
            u.energy_used += used
            comp["energy"] += used
            self.m["energy"] += used
            if u.energy <= 0.0:
                u.energy = 0.0
                self._release_commitment(u, t, comp, switch=False)
                self._fail(u, "energy_depleted")
            elif e.failure_rate > 0 and self.rng_dyn.random() < e.failure_rate:
                self._fail(u, "fault")
            elif u.status == UAVStatus.EN_ROUTE and self.cfg.safety.energy_filter:
                self._recheck_en_route(u, t, comp)

    def _recheck_en_route(self, u: UAV, t: int, comp) -> None:
        """Re-verify energy feasibility each step (wind can make the nominal estimate optimistic)."""
        task = self.tasks[u.task_id]
        e, s = self.cfg.env, self.cfg.safety
        d = float(np.linalg.norm(task.pos - u.pos))
        _, d_back = self.checker.nearest_base(task.pos)
        need = nominal_travel_energy(d, e) + service_energy(task.service_duration, e) + nominal_travel_energy(d_back, e)
        if u.energy - s.reserve_frac * u.capacity < need:
            self.m["energy_aborts"] += 1
            self._release_commitment(u, t, comp, switch=False)
            self._do_return_or_emergency(u, t)

    def _arrive(self, u: UAV, task: Task, t: int, comp) -> None:
        if task.status == TaskStatus.IN_PROGRESS and task.owner != u.id or task.status.terminal:
            self._abort_duplicate(u, t, comp, task)
            return
        u.status = UAVStatus.EXECUTING
        u.progress = 0.0
        task.status, task.owner = TaskStatus.IN_PROGRESS, u.id
        if self.mode != "centralized":
            self.coord.set_executing(u.id, t)

    def _abort_duplicate(self, u: UAV, t: int, comp, task: Task) -> None:
        self.m["duplicates"] += 1
        comp["duplicate"] += 1
        self._release_commitment(u, t, comp, switch=False, completed=task.status == TaskStatus.COMPLETED)
        u.status = UAVStatus.IDLE

    def _complete(self, u: UAV, task: Task, t: int, comp) -> None:
        task.status, task.completion_time = TaskStatus.COMPLETED, float(t + 1)
        comp["value"] += task.value
        self.m["completed_value"] += task.value
        self.m["completed"] += 1
        if task.feasible_ever:
            self.m["completed_feasible"] += 1
        self.last_completion_t = t + 1
        u.tasks_done += 1
        u.payload_rem = max(0.0, u.payload_rem - task.payload)
        self._release_commitment(u, t, comp, switch=False, completed=True)
        u.status = UAVStatus.IDLE

    def _expire_tasks(self, comp) -> None:
        for task in self.tasks:
            if task.announced and not task.status.terminal and self.t >= task.deadline:
                task.status = TaskStatus.EXPIRED
                comp["expired_value"] += task.value
                self.m["expired"] += 1
                self.m["expired_value"] += task.value
                for u in self.uavs:
                    if u.task_id == task.id and u.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE, UAVStatus.EXECUTING):
                        self._release_commitment(u, self.t, comp, switch=False)
                        u.status = UAVStatus.IDLE

    # ------------------------------------------------------------------ start of step
    def _begin_step(self, comp) -> None:
        t, e = self.t, self.cfg.env
        for task in self.tasks:
            if not task.announced and task.arrival_time <= t:
                task.announced = True
                if self.mode != "centralized":
                    self.coord.announce(TaskInfo.from_task(task), t)
        alive = np.array([u.status != UAVStatus.FAILED for u in self.uavs])
        self.comm.update_outages(t, alive)
        for u in self.uavs:
            u.comm_outage_until = int(self.comm.outage_until[u.id])
        if self.mode != "centralized":
            pos = np.stack([u.pos for u in self.uavs])
            self.coord.begin_step(t, pos, alive)
            events = self.coord.deliver(t, alive)
            events += self.coord.tick(t, self.comm.outage_until, alive)
            for uid, name, tid in events:
                self._handle_event(uid, name, tid, t, comp)
        self._refresh_truth()
        for u in self.uavs:
            if u.status in (UAVStatus.IDLE, UAVStatus.RESERVED) and self.checker.must_return(u):
                _, d = self.checker.nearest_base(u.pos)
                if u.status == UAVStatus.RESERVED:
                    self._release_commitment(u, t, comp, switch=False)
                if d > AT_BASE_RADIUS or u.soc < 0.9:
                    self.m["forced_returns"] += 1
                    self._do_return_or_emergency(u, t)

    def _handle_event(self, uid: int, name: str, tid: int, t: int, comp) -> None:
        u = self.uavs[uid]
        if u.status == UAVStatus.FAILED:
            return
        if name == "confirmed":
            if u.status == UAVStatus.RESERVED and u.task_id == tid:
                u.status = UAVStatus.EN_ROUTE
                self._note_commit(self.tasks[tid], comp)
        elif name in ("yield", "abandon"):
            if name == "abandon":
                self._release_commitment(u, t, comp, switch=False)
                if u.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE):
                    u.status = UAVStatus.IDLE
                return
            if u.task_id == tid and u.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE, UAVStatus.EXECUTING):
                if u.status in (UAVStatus.EN_ROUTE, UAVStatus.EXECUTING):
                    self.m["duplicates"] += 1
                    comp["duplicate"] += 1
                u.task_id, u.progress, u.status = -1, 0.0, UAVStatus.IDLE
                self.m["yields"] += 1
                self.last_conflict_t = t

    def _refresh_truth(self) -> None:
        by_task: Dict[int, List[UAV]] = defaultdict(list)
        for u in self.uavs:
            if u.task_id >= 0 and u.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE, UAVStatus.EXECUTING):
                by_task[u.task_id].append(u)
        rank = {UAVStatus.EXECUTING: 3, UAVStatus.EN_ROUTE: 2, UAVStatus.RESERVED: 1}
        smap = {3: TaskStatus.IN_PROGRESS, 2: TaskStatus.ASSIGNED, 1: TaskStatus.RESERVED}
        for task in self.tasks:
            if task.status.terminal:
                continue
            us = by_task.get(task.id)
            if not us:
                if task.status != TaskStatus.PENDING:
                    task.progress = 0.0
                task.status, task.owner = TaskStatus.PENDING, -1
                continue
            best = max(us, key=lambda x: (rank[x.status], -x.id))
            task.status, task.owner = smap[rank[best.status]], best.id

    # ------------------------------------------------------------------ reward / termination
    def _reward(self, comp) -> float:
        w, e = self.cfg.reward, self.cfg.env
        open_tasks = any((not x.status.terminal) for x in self.tasks if x.announced) or any(not x.announced for x in self.tasks)
        terms = {
            "r_value": w.w_c * comp["value"] / self.mean_value,
            "r_energy": -w.w_e * comp["energy"] / e.capacity,
            "r_time": -w.w_t * (e.dt / e.horizon) * (1.0 if open_tasks else 0.0),
            "r_deadline": -w.w_l * comp["expired_value"] / self.mean_value,
            "r_switch": -w.w_s * comp["switch"],
            "r_duplicate": -w.w_d * comp["duplicate"],
            "r_comm": -w.w_q * (self.comm.step_bytes / 1000.0) / e.n_uavs,
        }
        for k, v in terms.items():
            comp[k] = v
        return float(sum(terms.values()))

    def _mission_over(self) -> bool:
        if all(u.status == UAVStatus.FAILED for u in self.uavs):
            return True
        return all(x.status.terminal for x in self.tasks) and all(x.arrival_time <= self.t for x in self.tasks)

    # ------------------------------------------------------------------ metrics
    def summary(self) -> Dict[str, float]:
        e, m = self.cfg.env, self.m
        n = max(len(self.tasks), 1)
        total_value = sum(x.value for x in self.tasks) or 1.0
        n_feas = sum(1 for x in self.tasks if x.feasible_ever)
        minutes = max(self.t * e.dt / 60.0, 1e-9)
        done = m["completed"]
        lat = self.coord.metrics["reservation_latencies"] if self.mode != "centralized" else []
        part_rec = np.nan
        if self.comm.partition is not None:
            end = self.comm.partition[1]
            part_rec = float(max(self.last_conflict_t - end, 0)) if self.last_conflict_t >= end else 0.0
        out = {
            "weighted_completion_rate": m["completed_value"] / total_value,
            "total_completion_rate": done / n,
            "feasible_completion_rate": m["completed_feasible"] / max(n_feas, 1),
            "makespan": float(self.last_completion_t if done else self.t),
            "energy": m["energy"],
            "energy_per_task": m["energy"] / done if done else np.nan,
            "deadline_success_rate": done / max(done + m["expired"], 1),
            "deadline_violations": m["expired"],
            "duplicate_rate": m["duplicates"] / max(m["commitments"], 1),
            "duplicates": m["duplicates"],
            "reassignment_rate": m["reassignments"] / n,
            "reservation_latency": float(np.mean(lat)) if len(lat) else np.nan,
            "msgs_per_uav_min": self.comm.stats["messages"] / e.n_uavs / minutes,
            "bytes_per_uav_min": self.comm.stats["bytes"] / e.n_uavs / minutes,
            "conflicts": self.coord.metrics["conflicts"],
            "yields": m["yields"],
            "task_switches": m["task_switches"],
            "uav_failures": m["uav_failures"],
            "energy_depleted": m["energy_depleted"],
            "emergency_landings": m["emergency_landings"],
            "safety_overrides": m["safety_overrides"],
            "partition_recovery_steps": part_rec,
            "episode_return": self.ep_return,
            "steps": float(self.t),
            "n_uavs": float(e.n_uavs),
            "n_tasks": float(len(self.tasks)),
        }
        for k, v in self.comp_totals.items():
            if k.startswith("r_"):
                out["ret_" + k[2:]] = v
        return out
