"""Simplified kinematics, energy and wind models.

Simplifications (documented in docs/architecture.md):
  * 2-D point-mass UAVs flying straight lines at a constant air speed; no collision model.
  * Wind is a slowly varying Ornstein-Uhlenbeck vector; it changes the *ground speed along the path*
    (head/tail wind component) but not the heading. Energy is drawn per second flown, so head-winds
    cost extra energy per metre. Planning uses nominal (wind-free) estimates plus safety margins.
  * Flight control is separated from allocation: ``move_towards`` is the only place positions change.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import EnvConfig


class WindModel:
    def __init__(self, cfg: EnvConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.vec = rng.normal(0.0, cfg.wind_std, size=2) if cfg.wind_std > 0 else np.zeros(2)

    def step(self, dt: float) -> np.ndarray:
        c = self.cfg
        if c.wind_std <= 0:
            return self.vec
        a = np.exp(-dt / max(c.wind_tau, 1e-6))
        self.vec = a * self.vec + np.sqrt(1 - a * a) * c.wind_std * self.rng.normal(size=2)
        return self.vec


def nominal_travel_time(dist: np.ndarray | float, speed: float) -> np.ndarray | float:
    return dist / max(speed, 1e-6)


def nominal_travel_energy(dist: np.ndarray | float, cfg: EnvConfig) -> np.ndarray | float:
    return dist * cfg.move_energy_per_m


def service_energy(duration: np.ndarray | float, cfg: EnvConfig) -> np.ndarray | float:
    return duration * cfg.service_energy_per_s


def move_towards(pos: np.ndarray, target: np.ndarray, speed: float, wind: np.ndarray, dt: float,
                 cfg: EnvConfig) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Advance one step toward ``target``.

    Returns (new_pos, distance_moved, energy_used, velocity). Energy is charged for the full step flown
    (``move_energy_per_m * nominal_speed * dt``), scaled down if the UAV arrives early.
    """
    delta = target - pos
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9:
        return pos.copy(), 0.0, 0.0, np.zeros(2)
    d = delta / dist
    ground_speed = max(0.3 * speed, speed + float(np.dot(wind, d)))
    step = ground_speed * dt
    frac = 1.0
    if step >= dist:
        frac = dist / step
        step = dist
    energy = cfg.move_energy_per_m * speed * dt * frac
    return pos + d * step, step, energy, d * ground_speed


def segment_circle_blocked(a: np.ndarray, b: np.ndarray, zones: np.ndarray) -> np.ndarray:
    """Whether segment a->b intersects any circle ``zones[:, :3] = (x, y, r)``. Returns bool scalar array."""
    if zones.size == 0:
        return np.zeros((), dtype=bool)
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-12
    t = np.clip(((zones[:, :2] - a) @ ab) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.any(np.linalg.norm(closest - zones[:, :2], axis=1) < zones[:, 2])
