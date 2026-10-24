"""Local observation container, batching/collation, critic-only global state, and the decentralized heuristic expert."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..environment.task import TaskStatus
from ..environment.uav import UAVStatus
from .features import Dims, Norm, onehot

LOCAL_KEYS = ["uav_x", "uav_mask", "task_x", "task_mask", "base_x", "base_mask",
              "e_uu", "adj_uu", "e_ut", "adj_ut", "e_ub", "adj_ub", "action_mask"]


@dataclass
class LocalObservation:
    """Everything one UAV's actor may see. Built only from its own state and its local reservation table
    (which is fed exclusively by messages that actually arrived)."""
    ego_id: int
    t: int
    uav_x: np.ndarray        # [Nu, Fu]   index 0 = ego
    uav_mask: np.ndarray     # [Nu] bool
    task_x: np.ndarray       # [Nt, Ft]
    task_mask: np.ndarray    # [Nt] bool
    base_x: np.ndarray       # [Nb, Fb]
    base_mask: np.ndarray    # [Nb] bool
    e_uu: np.ndarray         # [Nu, Nu, Euu]
    adj_uu: np.ndarray       # [Nu, Nu] bool
    e_ut: np.ndarray         # [Nu, Nt, Eut]
    adj_ut: np.ndarray       # [Nu, Nt] bool
    e_ub: np.ndarray         # [Nu, Nb, Eub]
    adj_ub: np.ndarray       # [Nu, Nb] bool
    action_mask: np.ndarray  # [Nt + 3] bool: task slots, WAIT, RETURN, CONTINUE
    task_ids: np.ndarray     # [Nt] int (-1 = padding)
    n_neighbors: int = 0
    n_tasks: int = 0


def collate(obs: List[LocalObservation], device=None) -> Dict[str, "torch.Tensor"]:  # noqa: F821
    import torch

    out = {}
    for k in LOCAL_KEYS:
        arr = np.stack([getattr(o, k) for o in obs])
        t = torch.from_numpy(arr)
        out[k] = t.to(device) if device is not None else t
    return out


def batch_to(batch: Dict[str, "torch.Tensor"], device) -> Dict[str, "torch.Tensor"]:  # noqa: F821
    return {k: v.to(device) for k, v in batch.items()}


# ----------------------------------------------------------------------------- global (critic-only) state
def build_global_state(env) -> Dict[str, np.ndarray]:
    """Privileged ground-truth state used only by the centralized critic during training."""
    from ..environment.simulator import MAX_GLOBAL_TASK, MAX_GLOBAL_UAV

    cfg, e = env.cfg, env.cfg.env
    K = e.n_capabilities
    nm = Norm.from_cfg(cfg)
    Fu, Ft = 16 + K, 15 + K
    gu = np.zeros((MAX_GLOBAL_UAV, Fu), dtype=np.float32)
    gum = np.zeros(MAX_GLOBAL_UAV, dtype=bool)
    for u in env.uavs:
        f = np.concatenate([u.pos / nm.arena, u.vel / nm.speed_max, [u.soc, u.payload_rem / nm.payload_cap],
                            onehot(int(u.status), 7), [float(u.task_id >= 0)], u.capabilities.astype(np.float32),
                            [u.speed / nm.speed_max, float(u.comm_ok(env.t))]])
        gu[u.id], gum[u.id] = f, True
    gt = np.zeros((MAX_GLOBAL_TASK, Ft), dtype=np.float32)
    gtm = np.zeros(MAX_GLOBAL_TASK, dtype=bool)
    for x in env.tasks:
        if not x.announced:
            continue
        f = np.concatenate([x.pos / nm.arena, [x.priority / 5.0, np.clip((x.deadline - env.t) / nm.horizon, -1, 1),
                                               x.service_duration / nm.svc_max], onehot(x.required_capability, K),
                            [x.payload / nm.payload_cap], onehot(int(x.status), 7), [1.0, float(x.owner >= 0)]])
        gt[x.id], gtm[x.id] = f, True
    done = sum(1 for x in env.tasks if x.status == TaskStatus.COMPLETED)
    exp = sum(1 for x in env.tasks if x.status == TaskStatus.EXPIRED)
    n = max(len(env.tasks), 1)
    sc = np.array([env.t / nm.horizon, done / n, exp / n, sum(1 for x in env.tasks if not x.announced) / n,
                   sum(1 for u in env.uavs if u.status != UAVStatus.FAILED) / max(len(env.uavs), 1), e.n_uavs / 50.0],
                  dtype=np.float32)
    return {"g_uav": gu, "g_uav_mask": gum, "g_task": gt, "g_task_mask": gtm, "g_scalar": sc}


def collate_global(states: List[Dict[str, np.ndarray]], device=None):
    import torch

    out = {k: torch.from_numpy(np.stack([s[k] for s in states])) for k in states[0]}
    return {k: v.to(device) for k, v in out.items()} if device is not None else out


# ----------------------------------------------------------------------------- decentralized heuristic expert
class LocalGreedyPolicy:
    """Decentralized heuristic: every UAV picks the valid task with the highest *local* bid, using only its
    LocalObservation. Uses the same observation, protocol and safety layer as the learned policy. Also serves as
    the behaviour-cloning expert."""

    name = "local_greedy"
    centralized = False

    def __init__(self, cfg, dims: Dims):
        from ..coordination.bidding import compute_bid  # noqa: F401

        self.nm, self.dims, self.cfg = Norm.from_cfg(cfg), dims, cfg

    def bids(self, o: LocalObservation) -> np.ndarray:
        from ..coordination.bidding import compute_bid

        nm = self.nm
        b = np.full(self.dims.n_task, -np.inf)
        for j in np.nonzero(o.action_mask[: self.dims.n_task])[0]:
            tx, ee = o.task_x[j], o.e_ut[0, j]
            eta = ee[0] * nm.horizon
            svc = tx[4] * nm.svc_max
            slack = tx[3] * nm.horizon - eta - svc
            b[j] = compute_bid(tx[2] * 5.0, eta, svc, slack, ee[1] * nm.capacity, nm.capacity)
        return b

    def act_index(self, o: LocalObservation) -> int:
        nT = self.dims.n_task
        if o.action_mask[nT + 2]:          # holds a commitment -> keep it
            return nT + 2
        b = self.bids(o)
        if np.isfinite(b).any():
            return int(np.argmax(b))
        return nT                           # WAIT
