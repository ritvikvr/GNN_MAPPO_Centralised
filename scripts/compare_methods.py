#!/usr/bin/env python
"""Compare methods under reliable / degraded / failure conditions + scaling.  python scripts/compare_methods.py --config configs/evaluation.yaml"""
import argparse
import _common  # noqa: F401
from uav_allocation.config import load_config
from uav_allocation.evaluation.compare import run_comparison

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/evaluation.yaml")
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--episodes", type=int, default=None)
ap.add_argument("--no-scaling", action="store_true")
ap.add_argument("--scaling-only", action="store_true", help="skip the condition comparison")
ap.add_argument("--methods", nargs="*", default=None)
ap.add_argument("-o", "--override", action="append", default=[])
a = ap.parse_args()
cfg = load_config(a.config, overrides=a.override)
methods = a.methods or list(cfg.eval.methods)
ck = a.checkpoint or cfg.eval.checkpoint
import os
if "gnn_mappo" in methods and not os.path.exists(ck):
    print(f"[warn] checkpoint {ck} not found: skipping gnn_mappo"); methods.remove("gnn_mappo")
out = run_comparison(cfg, methods, ck, a.episodes or cfg.eval.n_episodes, cfg.eval.out_dir, conditions=({"reliable": {}} if a.scaling_only else (cfg.eval.conditions or None)), scaling=not a.no_scaling, conditions_enabled=not a.scaling_only)
print("results written to", out)
