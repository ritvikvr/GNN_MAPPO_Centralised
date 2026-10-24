"""UAV data model, operational states and the discrete action type."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class UAVStatus(IntEnum):
    IDLE = 0
    RESERVED = 1     # provisional claim held, waiting for confirmation
    EN_ROUTE = 2     # confirmed, flying to the task
    EXECUTING = 3
    RETURNING = 4
    CHARGING = 5
    FAILED = 6

    @property
    def available(self) -> bool:
        return self != UAVStatus.FAILED


N_ACTIVE_STATUS = 6  # one-hot size (FAILED nodes are masked out of graphs)


class ActionKind(IntEnum):
    TASK = 0
    WAIT = 1
    RETURN = 2
    CONTINUE = 3


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    task_id: int = -1


WAIT = Action(ActionKind.WAIT)
RETURN = Action(ActionKind.RETURN)
CONTINUE = Action(ActionKind.CONTINUE)


@dataclass
class UAV:
    id: int
    pos: np.ndarray
    speed: float
    capacity: float
    energy: float
    payload_capacity: float
    capabilities: np.ndarray            # bool [n_capabilities]
    payload_rem: float = 0.0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    status: UAVStatus = UAVStatus.IDLE
    task_id: int = -1                   # current task / reservation target (truth)
    base_id: int = -1                   # base currently targeted / charging at
    progress: float = 0.0               # mission progress of the current task in [0, 1]
    comm_outage_until: int = 0
    last_outage_end: int = 0
    fail_reason: str = ""
    # statistics
    energy_used: float = 0.0
    distance: float = 0.0
    tasks_done: int = 0
    switches: int = 0

    @property
    def soc(self) -> float:
        """Battery state of charge in [0, 1]."""
        return float(np.clip(self.energy / self.capacity, 0.0, 1.0))

    def comm_ok(self, t: int) -> bool:
        return self.status != UAVStatus.FAILED and t >= self.comm_outage_until

    @property
    def has_commitment(self) -> bool:
        return self.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE, UAVStatus.EXECUTING)
