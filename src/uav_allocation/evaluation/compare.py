"""Multi-method / multi-condition comparison, scaling study and plots. Everything is written to CSV first."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import Config, with_changes
from ..utils import get_logger
from ..visualization import plots
from .evaluate import evaluate_method, seeds_from
from .metrics import aggregate

log = get_logger("compare")


def run_comparison(cfg: Config, methods: List[str], checkpoint: Optional[str], n_episodes: int, out_dir: str,
                   seed_base: int = 0, conditions: Optional[dict] = None, scaling: bool = True, conditions_enabled: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds_from(seed_base, n_episodes)          # identical seeds for every method (paired comparison)
    conditions = conditions or {"reliable": {}}
    frames = []
    for cname, change in (conditions.items() if conditions_enabled else []):
        ccfg = with_changes(cfg, change)
        for m in methods:
            log.info("condition=%s method=%s", cname, m)
            frames.append(evaluate_method(m, ccfg, seeds, checkpoint, condition=cname))
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out / "episodes_conditions.csv", index=False)
        agg = aggregate(df, ["condition", "method"])
        agg.to_csv(out / "summary_conditions.csv", index=False)
        plots.baseline_bars(agg, out / "comparison_completion.png", "weighted_completion_rate", "condition")
        plots.baseline_bars(agg, out / "comparison_energy.png", "energy", "condition")
        if len(conditions) > 1:
            plots.robustness(agg, out / "comm_robustness.png")
    if scaling:
        fr = []
        for n in cfg.eval.fleet_sizes:
            scfg = with_changes(cfg, {"env": {"n_uavs": n, "n_tasks": int(round(n * cfg.eval.tasks_per_uav))}})
            for m in methods:
                d = evaluate_method(m, scfg, seeds, checkpoint, condition="scaling")
                d["fleet_size"] = n
                fr.append(d)
        sdf = pd.concat(fr, ignore_index=True)
        sdf.to_csv(out / "episodes_scaling.csv", index=False)
        sagg = aggregate(sdf, ["fleet_size", "method"])
        sagg.to_csv(out / "summary_scaling.csv", index=False)
        plots.scalability(sagg, out / "scalability.png")
    return out
