"""Evaluate any method (centralized baseline, local heuristic, or learned checkpoint) on held-out scenario seeds."""
from __future__ import annotations

import resource
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..algorithms.mappo import load_actor
from ..algorithms.rollout import ActorPolicy, run_episode
from ..baselines import CENTRALIZED
from ..config import Config, with_changes
from ..environment.simulator import UAVTaskEnv
from ..graph.features import feature_dims
from ..graph.observations import LocalGreedyPolicy
from ..utils import get_device, split_of, test_seed

CENTRALIZED_NAMES = set(CENTRALIZED)


def make_policy(method: str, cfg: Config, checkpoint: Optional[str] = None, device=None, time_single: bool = False):
    """Returns (policy, cfg_to_use). The learned actor's architecture comes from the checkpoint's saved config."""
    if method in CENTRALIZED:
        return CENTRALIZED[method](), cfg
    if method == "local_greedy":
        return LocalGreedyPolicy(cfg, feature_dims(cfg)), cfg
    if method == "gnn_mappo":
        if not checkpoint:
            raise ValueError("gnn_mappo needs --checkpoint")
        device = device or get_device("auto")
        actor, ck_cfg, dims = load_actor(checkpoint, device)
        # scenario/comm/env settings come from `cfg`; model/feature layout must match the checkpoint
        merged = with_changes(cfg, {"model": ck_cfg.to_dict()["model"]})
        return ActorPolicy(actor, dims, device, deterministic=True, time_single=time_single), merged
    raise ValueError(f"unknown method {method}")


def evaluate_method(method: str, cfg: Config, seeds: List[int], checkpoint: Optional[str] = None, condition: str = "default",
                    device=None, time_single: bool = False) -> pd.DataFrame:
    rows = []
    policy, cfg = make_policy(method, cfg, checkpoint, device, time_single)
    for s in seeds:
        assert split_of(s) == "test", "evaluation must use held-out test seeds"
        env = UAVTaskEnv(cfg)
        t0 = time.perf_counter()
        m = run_episode(env, policy, s)
        row: Dict[str, Any] = {"method": method, "condition": condition, "seed": s, **m}
        row["controller"] = "centralized" if method in CENTRALIZED else "decentralized"
        rows.append(row)
    df = pd.DataFrame(rows)
    if getattr(policy, "latencies_ms", None):
        lat = np.asarray(policy.latencies_ms)
        df.attrs["latency_ms_p50"], df.attrs["latency_ms_p95"] = float(np.median(lat)), float(np.percentile(lat, 95))
        df["infer_ms_p50"], df["infer_ms_p95"] = df.attrs["latency_ms_p50"], df.attrs["latency_ms_p95"]
    df["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return df


def seeds_from(base: int, n: int) -> List[int]:
    return [test_seed(base + i) for i in range(n)]
