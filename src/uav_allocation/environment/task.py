"""Task data model (ground truth) and the static task descriptor shared between UAVs."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class TaskStatus(IntEnum):
    PENDING = 0
    RESERVED = 1      # some UAV holds a provisional claim
    ASSIGNED = 2      # a UAV is committed and en route
    IN_PROGRESS = 3   # a UAV is executing
    COMPLETED = 4
    EXPIRED = 5
    FAILED = 6

    @property
    def terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.EXPIRED, TaskStatus.FAILED)


@dataclass
class Task:
    """Ground-truth task. ``required_capability`` doubles as the task type."""
    id: int
    pos: np.ndarray
    required_capability: int
    priority: int               # mission value in {1..5}
    arrival_time: float
    deadline: float
    service_duration: float
    payload: float
    status: TaskStatus = TaskStatus.PENDING
    owner: int = -1             # current reservation owner (highest-ranked committed UAV)
    version: int = 0            # reservation epoch (bumped on every release / reassignment)
    progress: float = 0.0
    feasible_ever: bool = True  # whether the task was achievable by *some* UAV from a full battery
    completion_time: float = -1.0
    announced: bool = False

    @property
    def task_type(self) -> int:
        return self.required_capability

    @property
    def value(self) -> float:
        return float(self.priority)


@dataclass(frozen=True)
class TaskInfo:
    """Immutable static descriptor broadcast to UAVs (what a UAV can *know* about a task)."""
    id: int
    x: float
    y: float
    required_capability: int
    priority: int
    arrival_time: float
    deadline: float
    service_duration: float
    payload: float

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @staticmethod
    def from_task(t: Task) -> "TaskInfo":
        return TaskInfo(t.id, float(t.pos[0]), float(t.pos[1]), t.required_capability, t.priority,
                        t.arrival_time, t.deadline, t.service_duration, t.payload)
