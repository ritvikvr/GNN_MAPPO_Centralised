"""Per-UAV lease-based reservation table (the *local view* of the shared task state).

Validity rules (see docs/architecture.md, "Reservation protocol")
----------------------------------------------------------------
A claim ``(task, uav, version, status)`` is:
  * PROVISIONAL after the UAV selects the task. It is *not* exclusive and the UAV does not move.
  * CONFIRMED once, at the owner, (a) ``confirm_window`` steps elapsed since the request, (b) no claim that outranks
    it was received in that window, and (c) the owner's radio was up for the whole window.
    Only a CONFIRMED claim lets the UAV leave for the task.
  * EXECUTING once the UAV arrived and started service (outranks everything).
  * Valid only while its *lease* (``expiry``) is live; owners renew every ``renew_period`` steps. A lease that is not
    renewed (owner failed / partitioned / out of range) simply lapses in the local view, so the task can be re-claimed
    (reassignment after UAV failure). Claims that arrive already expired are rejected as stale.
  * The task ``version`` (epoch) is bumped by an explicit RELEASE / abandon; messages carrying an older version are
    rejected as stale. Contention within one epoch is always settled by the deterministic order, never by arrival order.
Under partitions two UAVs can both CONFIRM the same task; the deterministic order in conflict_resolution.py makes
the loser yield as soon as the two claims meet again (eventual reconciliation, not instant consistency).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..environment.task import TaskInfo
from ..environment.uav import UAVStatus


class ResStatus(IntEnum):
    PROVISIONAL = 0
    CONFIRMED = 1
    EXECUTING = 2
    RELEASED = 3
    EXPIRED = 4

    @property
    def live(self) -> bool:
        return self in (ResStatus.PROVISIONAL, ResStatus.CONFIRMED, ResStatus.EXECUTING)


@dataclass(frozen=True)
class Reservation:
    task_id: int
    uav_id: int
    bid: float
    version: int          # task epoch this claim targets
    status: ResStatus
    timestamp: int        # step the claim (in this status) was issued
    expiry: int           # lease expiry step
    msg_id: int
    t_request: int = 0    # step the claim was first requested (latency bookkeeping)


class Outcome(IntEnum):
    ACCEPTED = 0
    RENEWED = 1
    STALE = 2
    DUPLICATE = 3
    KEPT = 4            # a rival claim arrived but our stored (or own) claim outranks it
    REPLACED = 5        # stored remote claim replaced by a higher-ranked remote claim
    LOST_OWN = 6        # our own claim was out-ranked -> caller must yield
    RELEASED = 7
    DONE = 8            # task already known complete / expired


@dataclass
class NeighborInfo:
    uav_id: int
    pos: np.ndarray
    vel: np.ndarray
    energy: float
    capacity: float
    payload_rem: float
    payload_capacity: float
    speed: float
    capabilities: np.ndarray
    status: int
    task_id: int
    sent_time: int
    recv_time: int


@dataclass
class UAVView:
    """Duck-typed UAV snapshot used by the feasibility checker for neighbours."""
    pos: np.ndarray
    energy: float
    capacity: float
    speed: float
    payload_rem: float
    capabilities: np.ndarray
    status: UAVStatus
    id: int = -1


class LocalReservationTable:
    """Everything one UAV knows about tasks, claims and neighbours. No global state."""

    def __init__(self, uav_id: int, lease_duration: int, confirm_window: int, renew_period: int):
        self.uav_id = uav_id
        self.lease, self.confirm_window, self.renew_period = lease_duration, confirm_window, renew_period
        self.tasks: Dict[int, TaskInfo] = {}
        self.claims: Dict[int, Reservation] = {}
        self.epochs: Dict[int, int] = {}
        self.done: Dict[int, int] = {}
        self.own: Optional[Reservation] = None
        self.neighbors: Dict[int, NeighborInfo] = {}
        self._seen: Dict[int, int] = {}
        self._last_renew = 0
        self.known_since: Dict[int, int] = {}

    # -------------------------------------------------------------- knowledge
    def epoch(self, tid: int) -> int:
        return self.epochs.get(tid, 0)

    def learn_task(self, info: TaskInfo, t: int) -> bool:
        if info.id in self.done or info.id in self.tasks:
            return False
        self.tasks[info.id] = info
        self.known_since[info.id] = t
        return True

    def mark_done(self, tid: int, t: int) -> None:
        self.done[tid] = t
        self.claims.pop(tid, None)
        if self.own is not None and self.own.task_id == tid:
            self.own = None

    def prune(self, t: int) -> None:
        """Drop expired tasks and stale neighbours; bounded memory."""
        for tid in [k for k, v in self.tasks.items() if t >= v.deadline]:
            self.tasks.pop(tid)
            self.done.setdefault(tid, t)
            self.claims.pop(tid, None)
        if len(self._seen) > 4096:
            for k in sorted(self._seen)[:2048]:
                self._seen.pop(k)

    def drop_stale_neighbors(self, t: int, timeout: int) -> None:
        for k in [k for k, v in self.neighbors.items() if t - v.recv_time > timeout]:
            self.neighbors.pop(k)

    # -------------------------------------------------------------- own claims
    def request(self, tid: int, bid: float, t: int, msg_id: int) -> Reservation:
        r = Reservation(tid, self.uav_id, float(bid), self.epoch(tid), ResStatus.PROVISIONAL, t, t + self.lease,
                        msg_id, t_request=t)
        self.claims[tid] = r
        self.own = r
        self._last_renew = t
        return r

    def commit_local(self, tid: int, bid: float, t: int, msg_id: int) -> Reservation:
        """Coordination disabled: local, instantly 'confirmed', never broadcast."""
        r = replace(self.request(tid, bid, t, msg_id), status=ResStatus.CONFIRMED)
        self.claims[tid] = r
        self.own = r
        return r

    def release_own(self, t: int, completed: bool, msg_id: int) -> Optional[Reservation]:
        """Release own claim. Returns the RELEASED record to broadcast and bumps the local epoch."""
        if self.own is None:
            return None
        o = self.own
        rec = replace(o, status=ResStatus.RELEASED, timestamp=t, msg_id=msg_id)
        self.claims.pop(o.task_id, None)
        self.epochs[o.task_id] = max(self.epoch(o.task_id), o.version + 1)
        if completed:
            self.mark_done(o.task_id, t)
        self.own = None
        return rec

    def set_executing(self, t: int, msg_id: int) -> Optional[Reservation]:
        if self.own is None:
            return None
        r = replace(self.own, status=ResStatus.EXECUTING, timestamp=t, expiry=t + self.lease, msg_id=msg_id)
        self.own = r
        self.claims[r.task_id] = r
        return r

    # -------------------------------------------------------------- incoming claims
    def receive_claim(self, res: Reservation, t: int) -> Outcome:
        from .conflict_resolution import winner  # local import: avoid cycle

        tid = res.task_id
        if res.uav_id == self.uav_id:
            return Outcome.DUPLICATE
        if tid in self.done:
            return Outcome.DONE
        if res.msg_id in self._seen and res.status.live:
            return Outcome.DUPLICATE
        self._seen[res.msg_id] = t
        cur_epoch = self.epoch(tid)
        # ---- release / expiry records
        if not res.status.live:
            if res.version < cur_epoch:
                return Outcome.STALE
            self.epochs[tid] = res.version + 1
            ex = self.claims.get(tid)
            if ex is not None and ex.version <= res.version and ex.uav_id != self.uav_id:
                self.claims.pop(tid)
            if self.own is not None and self.own.task_id == tid and self.own.version <= res.version:
                self.own = replace(self.own, version=res.version + 1)     # re-base our live claim onto the new epoch
                self.claims[tid] = self.own
            return Outcome.RELEASED
        # ---- live claims
        if res.version < cur_epoch:
            return Outcome.STALE
        if res.expiry < t:
            return Outcome.STALE            # lease already expired when it arrived
        if res.version > cur_epoch:
            self.epochs[tid] = res.version
            if self.own is not None and self.own.task_id == tid:
                self.own = replace(self.own, version=res.version)       # same contention rules apply on the new epoch
                self.claims[tid] = self.own
            else:
                self.claims.pop(tid, None)
        ex = self.claims.get(tid)
        if ex is None:
            self.claims[tid] = res
            return Outcome.ACCEPTED
        if ex.uav_id == res.uav_id:
            if (int(res.status), res.timestamp) >= (int(ex.status), ex.timestamp):
                self.claims[tid] = res
                return Outcome.RENEWED
            return Outcome.DUPLICATE
        # ---- genuine conflict between two different UAVs
        w = winner(ex, res)
        own_involved = self.own is not None and self.own.task_id == tid
        if w is res:
            self.claims[tid] = res
            if own_involved:
                self.own = None
                return Outcome.LOST_OWN
            return Outcome.REPLACED
        return Outcome.KEPT

    # -------------------------------------------------------------- periodic maintenance
    def tick(self, t: int, link_ok_since: int, msg_ids) -> Tuple[List[Reservation], List[Tuple[str, int]]]:
        """Confirm / renew own claim, expire remote leases.

        Returns (records to broadcast, events[(name, task_id)]).
        """
        out: List[Reservation] = []
        events: List[Tuple[str, int]] = []
        o = self.own
        if o is not None:
            if o.status == ResStatus.PROVISIONAL:
                if t - o.timestamp >= self.confirm_window and link_ok_since <= o.timestamp:
                    r = replace(o, status=ResStatus.CONFIRMED, timestamp=t, expiry=t + self.lease, msg_id=next(msg_ids))
                    self.own = r
                    self.claims[o.task_id] = r
                    out.append(r)
                    events.append(("confirmed", o.task_id))
                elif t >= o.expiry:   # never confirmed (e.g. radio kept failing): give up
                    events.append(("abandon", o.task_id))
            elif t - self._last_renew >= self.renew_period:
                r = replace(o, timestamp=t, expiry=t + self.lease, msg_id=next(msg_ids))
                self.own = r
                self.claims[o.task_id] = r
                self._last_renew = t
                out.append(r)
        for tid, c in list(self.claims.items()):
            if c.uav_id != self.uav_id and t > c.expiry:
                self.claims.pop(tid)           # lease lapsed locally; epoch unchanged (see module docstring)
                events.append(("lease_expired", tid))
        return out, events

    def live_claims(self) -> List[Reservation]:
        return [c for c in self.claims.values() if c.status.live]
