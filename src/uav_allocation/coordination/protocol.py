"""Glue between per-UAV reservation tables and the lossy network: the decentralized reservation protocol.

Message kinds used: ANNOUNCE (ground -> UAV, task descriptor), HEARTBEAT (state + gossip of live claims, completed
task ids and recently learned task descriptors), REQUEST (new/confirmed/renewed claim, immediate), RELEASE.
No UAV ever reads another UAV's table: all information crosses the network as `Message`s.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..config import CommConfig, CoordConfig
from ..environment.communication import CommNetwork, Message, MsgKind
from ..environment.task import TaskInfo
from .reservations import LocalReservationTable, NeighborInfo, Outcome, Reservation, ResStatus

Event = Tuple[int, str, int]   # (uav_id, name, task_id)


class CoordinationLayer:
    def __init__(self, n_uavs: int, cc: CoordConfig, mc: CommConfig, comm: CommNetwork):
        self.n, self.cc, self.mc, self.comm = n_uavs, cc, mc, comm
        self.tables = [LocalReservationTable(i, cc.lease_duration, cc.confirm_window, cc.renew_period)
                       for i in range(n_uavs)]
        self.adj = np.zeros((n_uavs, n_uavs), dtype=bool)
        self.metrics: Dict[str, Any] = defaultdict(float)
        self.metrics["reservation_latencies"] = []
        self.metrics["conflict_log"] = []          # (step, task_id, loser, winner)
        self._ids = itertools.count(1_000_000)    # claim msg ids (distinct from network ids)

    # ------------------------------------------------------------------ topology
    def begin_step(self, t: int, pos: np.ndarray, alive: np.ndarray) -> None:
        self.adj = self.comm.adjacency(pos, t, alive)

    def _nbrs(self, uid: int) -> np.ndarray:
        return np.nonzero(self.adj[uid])[0]

    # ------------------------------------------------------------------ announcements
    def announce(self, info: TaskInfo, t: int) -> None:
        for i in range(self.n):
            self.comm.ground_send(Message(MsgKind.ANNOUNCE, -1, {"task": info}, 48, t), i, t, self.mc.command_loss)

    # ------------------------------------------------------------------ own claims
    def request(self, uid: int, info: TaskInfo, bid: float, t: int) -> Reservation:
        tab = self.tables[uid]
        tab.learn_task(info, t)
        if not self.cc.enabled:
            return tab.commit_local(info.id, bid, t, next(self._ids))
        r = tab.request(info.id, bid, t, next(self._ids))
        self.metrics["requests"] += 1
        self._send_claim(uid, r, info, t)
        return r

    def _send_claim(self, uid: int, r: Reservation, info: TaskInfo | None, t: int, to: Sequence[int] | None = None) -> None:
        kind = MsgKind.REQUEST
        msg = Message(kind, uid, {"res": r, "task": info}, self.mc.claim_bytes + 40, t)
        self.comm.broadcast(msg, self._nbrs(uid) if to is None else to, t)

    def release(self, uid: int, t: int, completed: bool) -> None:
        tab = self.tables[uid]
        rec = tab.release_own(t, completed, next(self._ids))
        if rec is None:
            return
        self.metrics["releases"] += 1
        if not self.cc.enabled:
            return
        msg = Message(MsgKind.RELEASE, uid, {"res": rec, "completed": completed}, self.mc.claim_bytes, t)
        self.comm.broadcast(msg, self._nbrs(uid), t)

    def set_executing(self, uid: int, t: int) -> None:
        tab = self.tables[uid]
        r = tab.set_executing(t, next(self._ids))
        if r is not None and self.cc.enabled:
            self._send_claim(uid, r, tab.tasks.get(r.task_id), t)

    # ------------------------------------------------------------------ periodic work
    def tick(self, t: int, link_since: np.ndarray, alive: np.ndarray) -> List[Event]:
        events: List[Event] = []
        for i, tab in enumerate(self.tables):
            if not alive[i]:
                continue
            tab.prune(t)
            tab.drop_stale_neighbors(t, self.mc.neighbor_timeout)
            if not self.cc.enabled:
                continue
            recs, evs = tab.tick(t, int(link_since[i]), self._ids)
            fresh = {tid for name, tid in evs if name == "confirmed"}
            for r in recs:
                if r.task_id in fresh and r.status == ResStatus.CONFIRMED:
                    self.metrics["confirmed"] += 1
                    self.metrics["reservation_latencies"].append(t - r.t_request)
                self._send_claim(i, r, tab.tasks.get(r.task_id), t)
            for name, tid in evs:
                if name == "lease_expired":
                    self.metrics["lease_expired"] += 1
                events.append((i, name, tid))
        return events

    def send_heartbeats(self, t: int, uavs: Sequence[Any]) -> None:
        if not self.cc.enabled and self.mc.heartbeat_period <= 0:
            return
        for u in uavs:
            if not u.comm_ok(t):
                continue
            tab = self.tables[u.id]
            claims = tab.live_claims() if self.cc.enabled else []
            recent = [tab.tasks[k] for k, ts in tab.known_since.items() if k in tab.tasks and t - ts <= 10][:8]
            done = [k for k, ts in tab.done.items() if t - ts <= 40][:32]
            state = dict(pos=u.pos.copy(), vel=u.vel.copy(), energy=u.energy, capacity=u.capacity,
                         payload_rem=u.payload_rem, payload_capacity=u.payload_capacity, speed=u.speed,
                         capabilities=u.capabilities, status=int(u.status), task_id=tab.own.task_id if tab.own else -1)
            size = self.mc.heartbeat_bytes + self.mc.claim_bytes * len(claims) + 8 * len(done) + 40 * len(recent)
            msg = Message(MsgKind.HEARTBEAT, u.id, dict(state=state, claims=claims, done=done, tasks=recent), size, t)
            self.comm.broadcast(msg, self._nbrs(u.id), t)

    # ------------------------------------------------------------------ delivery
    def deliver(self, t: int, alive: np.ndarray) -> List[Event]:
        events: List[Event] = []
        for r, msg in self.comm.deliver(t):
            if not alive[r]:
                continue
            tab = self.tables[r]
            p = msg.payload
            if msg.kind == MsgKind.ANNOUNCE:
                tab.learn_task(p["task"], t)
            elif msg.kind == MsgKind.HEARTBEAT:
                s = p["state"]
                tab.neighbors[msg.sender] = NeighborInfo(
                    msg.sender, s["pos"], s["vel"], s["energy"], s["capacity"], s["payload_rem"],
                    s["payload_capacity"], s["speed"], s["capabilities"], s["status"], s["task_id"],
                    msg.send_time, t)
                for info in p["tasks"]:
                    tab.learn_task(info, t)
                for tid in p["done"]:
                    if tid not in tab.done:
                        tab.mark_done(tid, t)
                if self.cc.enabled:
                    for res in p["claims"]:
                        self._claim(r, res, t, events)
            elif self.cc.enabled and msg.kind in (MsgKind.REQUEST, MsgKind.RELEASE):
                if p.get("task") is not None:
                    tab.learn_task(p["task"], t)
                self._claim(r, p["res"], t, events)
                if msg.kind == MsgKind.RELEASE and p.get("completed"):
                    tab.mark_done(p["res"].task_id, t)
        return events

    def _claim(self, r: int, res: Reservation, t: int, events: List[Event]) -> None:
        tab = self.tables[r]
        mine_before = tab.own
        out = tab.receive_claim(res, t)
        if out == Outcome.STALE:
            self.metrics["stale_rejected"] += 1
        elif out == Outcome.LOST_OWN:
            self.metrics["conflicts"] += 1
            self.metrics["yields"] += 1
            events.append((r, "yield", res.task_id))
            if mine_before is not None:
                self.metrics["conflict_log"].append((t, res.task_id, r, res.uav_id))
        elif out == Outcome.KEPT and tab.own is not None and tab.own.task_id == res.task_id:
            self.metrics["conflicts"] += 1
            # defend: tell the rival we outrank it, right now (don't wait for the next heartbeat)
            self._send_claim(r, tab.own, tab.tasks.get(res.task_id), t, to=[res.uav_id])
            self.metrics["defends"] += 1
