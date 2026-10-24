"""Lossy, delayed, range-limited ad-hoc network model with outages and partitions.

Every message has a minimum latency of one simulation step; the configured ``delay_min..delay_max``
are *extra* steps. Packet loss is sampled independently for every (sender, receiver) pair.
"""
from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import CommConfig


class MsgKind(IntEnum):
    HEARTBEAT = 0      # state + gossip of live claims / completed tasks
    REQUEST = 1        # reservation request
    RELEASE = 2        # reservation released / task completed
    ANNOUNCE = 3       # ground-station task announcement
    COMMAND = 4        # centralized baseline command


@dataclass
class Message:
    kind: MsgKind
    sender: int                    # -1 = ground station
    payload: Dict[str, Any]
    size_bytes: int
    send_time: int
    msg_id: int = -1


class CommNetwork:
    def __init__(self, cfg: CommConfig, n_uavs: int, rng: np.random.Generator):
        self.cfg, self.n, self.rng = cfg, n_uavs, rng
        self._seq = itertools.count()
        self._msg_ids = itertools.count(1)
        self.queue: List[Tuple[int, int, int, Message]] = []
        self.outage_until = np.zeros(n_uavs, dtype=np.int64)
        self.partition: Optional[Tuple[int, int, float]] = None   # (start, end, x_split)
        self.stats: Dict[str, float] = defaultdict(float)
        self.bytes_by_kind: Dict[str, float] = defaultdict(float)
        self.step_bytes = 0.0

    # ------------------------------------------------------------------ topology
    def schedule_partition(self, start: int, duration: int, x_split: float) -> None:
        self.partition = (start, start + duration, x_split)

    def partition_active(self, t: int) -> bool:
        return self.partition is not None and self.partition[0] <= t < self.partition[1]

    def radio_up(self, t: int) -> np.ndarray:
        return t >= self.outage_until

    def update_outages(self, t: int, alive: np.ndarray) -> None:
        c = self.cfg
        if c.outage_prob <= 0:
            return
        start = (self.rng.random(self.n) < c.outage_prob) & alive & self.radio_up(t)
        for i in np.nonzero(start)[0]:
            self.outage_until[i] = t + int(self.rng.integers(c.outage_min, c.outage_max + 1))
            self.stats["outages"] += 1

    def adjacency(self, pos: np.ndarray, t: int, alive: np.ndarray) -> np.ndarray:
        """Boolean [n, n] symmetric link matrix (no self links)."""
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        up = self.radio_up(t) & alive
        adj = (d <= self.cfg.range) & up[:, None] & up[None, :]
        if self.partition_active(t):
            side = pos[:, 0] < self.partition[2]
            adj &= side[:, None] == side[None, :]
        np.fill_diagonal(adj, False)
        return adj

    @staticmethod
    def quality(dist: np.ndarray | float, rng_m: float) -> np.ndarray | float:
        return np.clip(1.0 - np.asarray(dist) / rng_m, 0.0, 1.0)

    # ------------------------------------------------------------------ transport
    def new_id(self) -> int:
        return next(self._msg_ids)

    def _push(self, receiver: int, msg: Message, t: int) -> None:
        c = self.cfg
        if self.rng.random() < c.loss_prob:
            self.stats["dropped"] += 1
            return
        delay = int(self.rng.integers(c.delay_min, c.delay_max + 1))
        heapq.heappush(self.queue, (t + 1 + delay, next(self._seq), receiver, msg))

    def broadcast(self, msg: Message, receivers: Sequence[int], t: int) -> None:
        if msg.msg_id < 0:
            msg.msg_id = self.new_id()
        for r in receivers:
            self._count(msg)
            self._push(int(r), msg, t)

    def send(self, msg: Message, receiver: int, t: int) -> None:
        self.broadcast(msg, [receiver], t)

    def ground_send(self, msg: Message, receiver: int, t: int, loss: float) -> bool:
        """Ground-station -> UAV link (announcements/commands). Returns False if lost/outage now."""
        if msg.msg_id < 0:
            msg.msg_id = self.new_id()
        self._count(msg)
        if t < self.outage_until[receiver] or self.rng.random() < loss:
            self.stats["dropped"] += 1
            return False
        delay = int(self.rng.integers(self.cfg.delay_min, self.cfg.delay_max + 1))
        heapq.heappush(self.queue, (t + 1 + delay, next(self._seq), receiver, msg))
        return True

    def deliver(self, t: int) -> List[Tuple[int, Message]]:
        out: List[Tuple[int, Message]] = []
        while self.queue and self.queue[0][0] <= t:
            _, _, r, m = heapq.heappop(self.queue)
            if t < self.outage_until[r]:
                self.stats["dropped_outage"] += 1
                continue
            self.stats["delivered"] += 1
            out.append((r, m))
        return out

    def _count(self, msg: Message) -> None:
        self.stats["messages"] += 1
        self.stats["bytes"] += msg.size_bytes
        self.bytes_by_kind[msg.kind.name] += msg.size_bytes
        self.step_bytes += msg.size_bytes
