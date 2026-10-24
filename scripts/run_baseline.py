#!/usr/bin/env python
"""Run one episode of a baseline / heuristic.  python scripts/run_baseline.py --method greedy --uavs 20 --tasks 50 --seed 42"""
import argparse, json
import _common  # noqa: F401
from uav_allocation.config import load_config
from uav_allocation.environment.simulator import UAVTaskEnv
from uav_allocation.algorithms.rollout import run_episode
from uav_allocation.evaluation.evaluate import make_policy
from uav_allocation.utils import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="greedy", choices=["nearest", "greedy", "hungarian", "local_greedy"])
ap.add_argument("--uavs", type=int, default=20)
ap.add_argument("--tasks", type=int, default=50)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--config", default=None)
ap.add_argument("-o", "--override", action="append", default=[])
a = ap.parse_args()
cfg = load_config(a.config, overrides=[f"env.n_uavs={a.uavs}", f"env.n_tasks={a.tasks}"] + a.override)
seed_everything(a.seed)
pol, cfg = make_policy(a.method, cfg)
m = run_episode(UAVTaskEnv(cfg), pol, a.seed)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=1))
