#!/usr/bin/env python
"""Train GNN-MAPPO.  Example:  python scripts/train.py --config configs/training.yaml --seed 42"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_allocation.algorithms.trainer import Trainer  # noqa: E402
from uav_allocation.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from (loads optimizer state + iteration)")
    ap.add_argument("--tag", default="run", help="name used for log / checkpoint files")
    ap.add_argument("--override", "-o", action="append", default=[], help="e.g. -o train.iterations=10 -o model.recurrent=true")
    a = ap.parse_args()
    cfg = load_config(a.config, overrides=a.override)
    best = Trainer(cfg, seed=a.seed, resume=a.resume, tag=a.tag).run()
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
