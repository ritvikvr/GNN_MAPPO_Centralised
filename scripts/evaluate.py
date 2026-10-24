#!/usr/bin/env python
"""Evaluate a checkpoint on held-out TEST seeds.  python scripts/evaluate.py --checkpoint checkpoints/best.pt --uavs 20 --tasks 50 --seed 100"""
import argparse
from pathlib import Path
import _common  # noqa: F401
from uav_allocation.config import load_config
from uav_allocation.evaluation.evaluate import evaluate_method, seeds_from
from uav_allocation.evaluation.metrics import aggregate

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/best.pt")
ap.add_argument("--method", default="gnn_mappo")
ap.add_argument("--uavs", type=int, default=20)
ap.add_argument("--tasks", type=int, default=50)
ap.add_argument("--seed", type=int, default=100, help="first test-scenario index (test seeds are offset so they never overlap training)")
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--config", default=None)
ap.add_argument("--out", default="results/eval.csv")
ap.add_argument("-o", "--override", action="append", default=[])
a = ap.parse_args()
cfg = load_config(a.config, overrides=[f"env.n_uavs={a.uavs}", f"env.n_tasks={a.tasks}"] + a.override)
df = evaluate_method(a.method, cfg, seeds_from(a.seed, a.episodes), a.checkpoint, time_single=(a.method == "gnn_mappo"))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
ag = aggregate(df, ["method"])
print(ag.T.to_string())
print("saved", a.out)
