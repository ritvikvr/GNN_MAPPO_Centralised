"""Hard feasibility / safety filter (mission-level only).

This module is *independent of the learned policy*: it decides which assignments may be proposed.
It never claims collision-free flight; flight-level collision avoidance is out of scope (see docs).

Rejection reasons are returned as a bit-mask so tests and logs can say *why* a task was rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..config import EnvConfig, SafetyConfig
from ..environment.dynamics import nominal_travel_energy, nominal_travel_time, segment_circle_blocked, service_energy
from ..environment.uav import UAVStatus

UNAVAILABLE, CAPABILITY, PAYLOAD, GEOFENCE, NOFLY, ENERGY, DEADLINE, UNREACHABLE = 1, 2, 4, 8, 16, 32, 64, 128
REASON_NAMES = {UNAVAILABLE: "unavailable", CAPABILITY: "capability", PAYLOAD: "payload", GEOFENCE: "geofence",
                NOFLY: "no_fly_path", ENERGY: "energy", DEADLINE: "deadline", UNREACHABLE: "unreachable"}


@dataclass
class TaskArrays:
    """Vectorised static task data (n tasks)."""
    pos: np.ndarray          # [n, 2]
    cap: np.ndarray          # [n] int
    payload: np.ndarray      # [n]
    service: np.ndarray      # [n]
    deadline: np.ndarray     # [n]
    priority: np.ndarray     # [n]

    @staticmethod
    def from_infos(infos: Sequence[Any]) -> "TaskArrays":
        n = len(infos)
        return TaskArrays(
            pos=np.array([[i.x, i.y] for i in infos], dtype=np.float64).reshape(n, 2),
            cap=np.array([i.required_capability for i in infos], dtype=np.int64),
            payload=np.array([i.payload for i in infos], dtype=np.float64),
            service=np.array([i.service_duration for i in infos], dtype=np.float64),
            deadline=np.array([i.deadline for i in infos], dtype=np.float64),
            priority=np.array([i.priority for i in infos], dtype=np.float64),
        )


@dataclass
class Feasibility:
    mask: np.ndarray        # bool [n]  True = assignment allowed
    reasons: np.ndarray     # int  [n]  bit-mask of violated checks (0 = ok)
    eta: np.ndarray         # nominal travel time to task (s)
    e_travel: np.ndarray    # nominal travel energy to task
    e_service: np.ndarray
    e_return: np.ndarray    # energy from task location to the nearest base
    margin: np.ndarray      # energy left after the task and return, minus reserve (can be negative)


class FeasibilityChecker:
    def __init__(self, env_cfg: EnvConfig, safety_cfg: SafetyConfig, bases: np.ndarray):
        self.env, self.safety = env_cfg, safety_cfg
        self.bases = np.asarray(bases, dtype=np.float64).reshape(-1, 2)
        self.zones = np.asarray(env_cfg.no_fly_zones, dtype=np.float64).reshape(-1, 3)

    # ----------------------------------------------------------------- energy helpers
    def reserve_energy(self, capacity: float) -> float:
        return self.safety.reserve_frac * capacity

    def nearest_base(self, pos: np.ndarray) -> tuple[int, float]:
        d = np.linalg.norm(self.bases - pos[None, :], axis=1)
        j = int(np.argmin(d))
        return j, float(d[j])

    def return_energy(self, pos: np.ndarray) -> float:
        """Estimated energy to reach the nearest recovery point (with margin)."""
        _, d = self.nearest_base(pos)
        return float(nominal_travel_energy(d, self.env)) * (1.0 + self.safety.energy_margin)

    def must_return(self, uav: Any) -> bool:
        """True when remaining energy only just covers the return trip plus the mandatory reserve."""
        if not self.safety.energy_filter:
            return False
        slack = uav.energy - self.return_energy(uav.pos) - self.reserve_energy(uav.capacity)
        return slack <= 0.0

    def can_return(self, uav: Any) -> bool:
        return uav.energy >= self.return_energy(uav.pos)

    # ----------------------------------------------------------------- main check
    def check(self, uav: Any, tasks: TaskArrays, now: float) -> Feasibility:
        n = len(tasks.cap)
        env, s = self.env, self.safety
        reasons = np.zeros(n, dtype=np.int64)
        if n == 0:
            z = np.zeros(0)
            return Feasibility(np.zeros(0, dtype=bool), reasons, z, z, z, z, z)
        if uav.status in (UAVStatus.FAILED, UAVStatus.CHARGING, UAVStatus.RETURNING):
            reasons |= UNAVAILABLE
        reasons |= np.where(~np.asarray(uav.capabilities)[tasks.cap], CAPABILITY, 0)
        reasons |= np.where(tasks.payload > uav.payload_rem + 1e-9, PAYLOAD, 0)
        inside = np.all((tasks.pos >= 0.0) & (tasks.pos <= env.arena), axis=1)
        reasons |= np.where(~inside, GEOFENCE, 0)
        if len(self.zones):
            for k in range(n):
                if segment_circle_blocked(uav.pos, tasks.pos[k], self.zones):
                    reasons[k] |= NOFLY
        dist = np.linalg.norm(tasks.pos - uav.pos[None, :], axis=1)
        eta = np.asarray(nominal_travel_time(dist, uav.speed))
        e_tr = np.asarray(nominal_travel_energy(dist, env))
        e_sv = np.asarray(service_energy(tasks.service, env))
        d_back = np.linalg.norm(tasks.pos[:, None, :] - self.bases[None, :, :], axis=2).min(axis=1)
        e_back = np.asarray(nominal_travel_energy(d_back, env)) * (1.0 + s.energy_margin)
        need = (e_tr * (1.0 + s.energy_margin)) + e_sv + e_back
        usable = uav.energy - self.reserve_energy(uav.capacity)
        margin = usable - need
        if s.energy_filter:
            reasons |= np.where(e_tr * (1.0 + s.energy_margin) > usable, UNREACHABLE, 0)
            reasons |= np.where(margin < 0.0, ENERGY, 0)
        if env.hard_deadlines:
            finish = now + eta * (1.0 + s.time_margin) + tasks.service
            reasons |= np.where(finish > tasks.deadline, DEADLINE, 0)
        return Feasibility(reasons == 0, reasons, eta, e_tr, e_sv, e_back, margin)

    def explain(self, reasons: int) -> list[str]:
        return [name for bit, name in REASON_NAMES.items() if reasons & bit]
