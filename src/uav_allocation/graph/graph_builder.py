"""Builds each UAV's *local* heterogeneous graph from information that UAV actually has.

Information sources for UAV i: (1) its own true state, (2) its local reservation table (tasks announced to it or learned
by gossip, claims heard, completed-task ids) and (3) neighbour states from heartbeats that actually arrived (with age).
It never reads other UAVs' true state, true task status, other tables, or future arrivals.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..coordination.reservations import UAVView
from ..environment.dynamics import nominal_travel_energy
from ..environment.uav import N_ACTIVE_STATUS, UAVStatus
from ..safety.feasibility import FeasibilityChecker, TaskArrays
from .features import E_UB, E_UT, E_UU, Dims, Norm, feature_dims, onehot
from .observations import LocalObservation

AT_BASE = 5.0


class GraphBuilder:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg
        self.dims: Dims = feature_dims(env.cfg)
        self.nm = Norm.from_cfg(env.cfg)

    # ------------------------------------------------------------------ helpers
    def _status_onehot(self, status: int) -> np.ndarray:
        return onehot(int(status), N_ACTIVE_STATUS)

    def _uav_feat(self, pos, vel, speed, soc, energy, capacity, payload_rem, payload_cap, status, committed, caps, age, ego) -> np.ndarray:
        nm, ck = self.nm, self.env.checker
        usable = max(0.0, energy - ck.reserve_energy(capacity)) / nm.capacity
        ret = ck.return_energy(pos) / nm.capacity
        return np.concatenate([pos / nm.arena, vel / nm.speed_max, [speed / nm.speed_max, soc, usable,
                                payload_rem / nm.payload_cap], self._status_onehot(status),
                               [float(committed), min(ret, 2.0)], caps.astype(np.float32),
                               [min(age / nm.neighbor_timeout, 1.0), float(ego)]]).astype(np.float32)

    # ------------------------------------------------------------------ main
    def build(self, uid: int) -> LocalObservation:
        env, d, nm, cfg = self.env, self.dims, self.nm, self.cfg
        t = env.t
        u = env.uavs[uid]
        tab = env.coord.tables[uid]
        ck: FeasibilityChecker = env.checker
        coord_on = cfg.coordination.enabled
        K = cfg.env.n_capabilities
        # ------------------------------------------------ candidate tasks from the local table
        known = [i for tid, i in tab.tasks.items() if tid not in tab.done and i.deadline > t]
        arr = TaskArrays.from_infos(known)
        feas = ck.check(u, arr, t)
        own = tab.own
        own_tid = own.task_id if own is not None else -1

        def claimed_by_other(tid: int) -> Optional[object]:
            c = tab.claims.get(tid)
            if c is not None and c.uav_id != uid and c.status.live and t <= c.expiry:
                return c
            return None

        order = self._select_tasks(known, feas, arr, own_tid)
        nT = len(order)
        # ------------------------------------------------ neighbour nodes
        nbrs = sorted(tab.neighbors.values(), key=lambda n: float(np.linalg.norm(n.pos - u.pos)))
        nbrs = [n for n in nbrs if t - n.recv_time <= cfg.comm.neighbor_timeout and n.status != int(UAVStatus.FAILED)][: d.n_uav - 1]
        nU = 1 + len(nbrs)
        # ------------------------------------------------ allocate padded arrays
        uav_x = np.zeros((d.n_uav, d.f_uav), np.float32)
        uav_mask = np.zeros(d.n_uav, bool)
        task_x = np.zeros((d.n_task, d.f_task), np.float32)
        task_mask = np.zeros(d.n_task, bool)
        base_x = np.zeros((d.n_base, d.f_base), np.float32)
        base_mask = np.ones(d.n_base, bool)
        e_uu = np.zeros((d.n_uav, d.n_uav, E_UU), np.float32)
        adj_uu = np.zeros((d.n_uav, d.n_uav), bool)
        e_ut = np.zeros((d.n_uav, d.n_task, E_UT), np.float32)
        adj_ut = np.zeros((d.n_uav, d.n_task), bool)
        e_ub = np.zeros((d.n_uav, d.n_base, E_UB), np.float32)
        adj_ub = np.zeros((d.n_uav, d.n_base), bool)
        action_mask = np.zeros(d.n_action, bool)
        task_ids = np.full(d.n_task, -1, np.int64)
        # ------------------------------------------------ UAV nodes
        views: List[UAVView] = [UAVView(u.pos, u.energy, u.capacity, u.speed, u.payload_rem, u.capabilities, u.status, u.id)]
        uav_x[0] = self._uav_feat(u.pos / 1.0, u.vel, u.speed, u.soc, u.energy, u.capacity, u.payload_rem, u.payload_capacity,
                                  int(u.status), own is not None or u.has_commitment, u.capabilities, 0.0, True)
        uav_mask[0] = True
        for k, n in enumerate(nbrs, start=1):
            age = t - n.sent_time
            soc = float(np.clip(n.energy / max(n.capacity, 1e-9), 0, 1))
            uav_x[k] = self._uav_feat(n.pos, n.vel, n.speed, soc, n.energy, n.capacity, n.payload_rem, n.payload_capacity,
                                      n.status, n.task_id >= 0, n.capabilities, age, False)
            uav_mask[k] = True
            views.append(UAVView(n.pos, n.energy, n.capacity, n.speed, n.payload_rem, n.capabilities, UAVStatus(n.status), n.uav_id))
        # ---- UAV-UAV edges (ego<->neighbour always; neighbour<->neighbour when within comm range)
        pos_all = np.stack([v.pos for v in views])
        dist_all = np.linalg.norm(pos_all[:, None] - pos_all[None], axis=-1)
        ages = np.array([0.0] + [t - n.sent_time for n in nbrs])
        for i in range(nU):
            for j in range(nU):
                linked = (i == j) or (i == 0) or (j == 0) or dist_all[i, j] <= cfg.comm.range
                if not linked:
                    continue
                adj_uu[i, j] = True
                q = float(np.clip(1.0 - dist_all[i, j] / nm.comm_range, 0.0, 1.0))
                e_uu[i, j] = [min(dist_all[i, j] / nm.comm_range, 2.0), q, min(max(ages[i], ages[j]) / nm.neighbor_timeout, 1.0), 1.0]
        # ------------------------------------------------ task nodes + UAV-TASK edges
        sel = np.array(order, dtype=np.int64)
        if nT:
            sub = TaskArrays(arr.pos[sel], arr.cap[sel], arr.payload[sel], arr.service[sel], arr.deadline[sel], arr.priority[sel])
            ego_f = (feas.eta[sel], feas.e_travel[sel], feas.mask[sel], feas.margin[sel], feas.reasons[sel])
            for s, j in enumerate(order):
                info = known[j]
                c = claimed_by_other(info.id)
                mine = info.id == own_tid
                state = np.array([c is None and not mine, c is not None, mine], np.float32)
                age = min((t - tab.known_since.get(info.id, t)) / 60.0, 1.0)
                unc = min(1.0, age + (c is not None) * 0.1 + cfg.env.arrival_uncertainty)
                task_x[s] = np.concatenate([info.pos / nm.arena, [info.priority / 5.0, np.clip((info.deadline - t) / nm.horizon, 0, 1),
                                            info.service_duration / nm.svc_max], onehot(info.required_capability, K),
                                            [info.payload / nm.payload_cap], state, [unc]]).astype(np.float32)
                task_mask[s] = True
                task_ids[s] = info.id
            for vi, v in enumerate(views):
                if vi == 0:
                    eta, e_tr, ok, margin, reasons = ego_f
                    compat = ((reasons & (2 | 4)) == 0)
                else:
                    f = ck.check(v, sub, t)
                    eta, e_tr, ok, margin = f.eta, f.e_travel, f.mask, f.margin
                    compat = ((f.reasons & (2 | 4)) == 0)
                for s in range(nT):
                    cl = tab.claims.get(known[order[s]].id)
                    is_owner = cl is not None and cl.uav_id == v.id and cl.status.live
                    other = cl is not None and cl.uav_id != v.id and cl.status.live and t <= cl.expiry
                    if vi == 0 and known[order[s]].id == own_tid:
                        is_owner, other = True, False
                    e_ut[vi, s] = [min(eta[s] / nm.horizon, 1.0), min(e_tr[s] / nm.capacity, 1.5), float(compat[s]), float(ok[s]),
                                   float(other), float(np.clip(margin[s] / nm.capacity, -1.0, 1.0)), float(is_owner)]
                    adj_ut[vi, s] = bool(compat[s])
        # ------------------------------------------------ bases
        _, nearest_d = ck.nearest_base(u.pos)
        nb_idx = int(np.argmin(np.linalg.norm(ck.bases - u.pos[None], axis=1)))
        for b in range(d.n_base):
            base_x[b] = [ck.bases[b, 0] / nm.arena, ck.bases[b, 1] / nm.arena, float(b == nb_idx)]
            for vi, v in enumerate(views):
                db = float(np.linalg.norm(ck.bases[b] - v.pos))
                et = float(nominal_travel_energy(db, cfg.env)) * (1 + cfg.safety.energy_margin)
                e_ub[vi, b] = [min(db / nm.arena, 1.5), min(et / nm.capacity, 1.5), float(v.energy >= et)]
                adj_ub[vi, b] = True
        # ------------------------------------------------ action mask (hard safety layer applied BEFORE the policy)
        status_ok = u.status in (UAVStatus.IDLE, UAVStatus.RESERVED, UAVStatus.EN_ROUTE)
        for s in range(nT):
            tid = known[order[s]].id
            if not status_ok or tid == own_tid or not feas.mask[order[s]]:
                continue
            c = claimed_by_other(tid)
            if coord_on and c is not None and int(c.status) >= 1:
                continue
            action_mask[s] = True
        committed = u.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE)
        forced = ck.must_return(u)
        action_mask[d.n_task + 0] = status_ok and not (forced and not committed)
        can_ret = ck.can_return(u) or cfg.safety.emergency_policy == "continue_to_base"
        action_mask[d.n_task + 1] = status_ok and can_ret and (nearest_d > AT_BASE or u.soc < 0.9)
        action_mask[d.n_task + 2] = committed and not forced
        if not action_mask.any():            # never leave an empty action set: hold is always the safe default
            action_mask[d.n_task + 0] = True
        return LocalObservation(uid, t, uav_x, uav_mask, task_x, task_mask, base_x, base_mask, e_uu, adj_uu, e_ut, adj_ut,
                                e_ub, adj_ub, action_mask, task_ids, n_neighbors=len(nbrs), n_tasks=nT)

    def _select_tasks(self, known, feas, arr: TaskArrays, own_tid: int) -> List[int]:
        """Prune to ``max_task_nodes`` slots: keep own task; then top-priority feasible tasks (so high-priority candidates
        survive pruning); then nearest feasible; then nearest infeasible ones as context."""
        n, cap = len(known), self.dims.n_task
        if n == 0:
            return []
        idx = list(range(n))
        chosen: List[int] = []
        for j in idx:
            if known[j].id == own_tid:
                chosen.append(j)
        rest = [j for j in idx if j not in chosen]
        feas_ids = [j for j in rest if feas.mask[j]]
        infeas = [j for j in rest if not feas.mask[j]]
        by_prio = sorted(feas_ids, key=lambda j: (-known[j].priority, feas.eta[j]))
        take = by_prio[: max(1, (cap - len(chosen) + 1) // 2)]
        chosen += take
        near = sorted([j for j in feas_ids if j not in take], key=lambda j: feas.eta[j])
        chosen += near[: cap - len(chosen)]
        if len(chosen) < cap:
            chosen += sorted(infeas, key=lambda j: feas.eta[j])[: cap - len(chosen)]
        return chosen[:cap]
