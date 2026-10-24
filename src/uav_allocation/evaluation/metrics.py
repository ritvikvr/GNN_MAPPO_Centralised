"""Aggregation of per-episode metrics: mean, std, 95% CI (Student t) and paired differences."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["weighted_completion_rate", "total_completion_rate", "feasible_completion_rate", "makespan", "energy",
           "energy_per_task", "deadline_success_rate", "deadline_violations", "duplicate_rate", "reassignment_rate",
           "reservation_latency", "msgs_per_uav_min", "bytes_per_uav_min", "episode_return"]


def ci95(x: Iterable[float]) -> float:
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(a) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def aggregate(df: pd.DataFrame, by: List[str], metrics: List[str] = METRICS) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n_episodes"] = len(g)
        for m in metrics:
            if m in g:
                v = g[m].astype(float)
                r[m + "_mean"], r[m + "_std"], r[m + "_ci95"] = v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan, ci95(v)
        rows.append(r)
    return pd.DataFrame(rows)


def paired_diff(df: pd.DataFrame, a: str, b: str, metric: str, by: str = "method", key: str = "seed") -> Dict[str, float]:
    """Mean paired difference (a - b) over identical scenario seeds with 95% CI and paired t-test p-value."""
    x = df[df[by] == a].set_index(key)[metric]
    y = df[df[by] == b].set_index(key)[metric]
    d = (x - y).dropna()
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 else float("nan")
    return {"mean_diff": float(d.mean()), "ci95": ci95(d), "p_value": p, "n": len(d)}
