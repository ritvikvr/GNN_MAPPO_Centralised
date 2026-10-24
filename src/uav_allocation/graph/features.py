"""Feature schemas, dimensions and normalisation constants for the heterogeneous graph.

Node types
  UAV  (F_u = 18 + K): pos(2) vel(2) speed soc usable_energy payload_frac status_onehot(6) has_commitment
                       return_energy cap_multihot(K) msg_age is_ego
  TASK (F_t = 10 + K): pos(2) priority ttd service cap_onehot(K) payload status3[free, claimed_by_other, mine]
                       uncertainty
  BASE (F_b = 3):      pos(2) is_nearest_to_ego
Edge types (all dense, masked by adjacency)
  UAV-UAV  (E_uu = 4): dist/range, link quality, message age, link flag
  UAV-TASK (E_ut = 7): eta, travel energy, compatibility(cap & payload), feasible(hard filter), reservation conflict,
                       recovery margin, owns-claim
  UAV-BASE (E_ub = 3): dist, travel energy, reachable
K = number of capability types.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config

N_BASE_FEAT = 3
E_UU, E_UT, E_UB = 4, 7, 3
N_SPECIAL = 3                      # WAIT, RETURN, CONTINUE
IDX_WAIT_OFF, IDX_RETURN_OFF, IDX_CONTINUE_OFF = 0, 1, 2   # offsets after the task slots


@dataclass(frozen=True)
class Dims:
    f_uav: int
    f_task: int
    f_base: int
    e_uu: int
    e_ut: int
    e_ub: int
    n_uav: int       # max UAV nodes (ego + neighbours)
    n_task: int      # max task slots
    n_base: int
    n_action: int    # n_task + N_SPECIAL


def feature_dims(cfg: Config) -> Dims:
    K = cfg.env.n_capabilities
    return Dims(18 + K, 10 + K, N_BASE_FEAT, E_UU, E_UT, E_UB, cfg.model.max_uav_nodes, cfg.model.max_task_nodes,
                cfg.env.n_bases, cfg.model.max_task_nodes + N_SPECIAL)


@dataclass(frozen=True)
class Norm:
    arena: float
    horizon: float
    capacity: float
    speed_max: float
    svc_max: float
    comm_range: float
    payload_cap: float
    neighbor_timeout: float

    @staticmethod
    def from_cfg(cfg: Config) -> "Norm":
        e = cfg.env
        return Norm(arena=e.arena, horizon=float(e.horizon), capacity=e.capacity, speed_max=e.speed * (1 + e.speed_hetero) * 1.5,
                    svc_max=float(max(e.service_range)), comm_range=cfg.comm.range, payload_cap=max(e.payload_capacity, 1e-6),
                    neighbor_timeout=float(max(cfg.comm.neighbor_timeout, 1)))


def onehot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= i < n:
        v[i] = 1.0
    return v


# global (critic-only) feature sizes
def global_dims(cfg: Config) -> tuple[int, int, int]:
    K = cfg.env.n_capabilities
    return 16 + K, 15 + K, 6
