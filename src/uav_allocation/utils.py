"""Seeding, scenario-seed partitioning, logging and device helpers."""
from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np

# Scenario seeds are partitioned so train / validation / test never overlap.
TRAIN_OFFSET, VAL_OFFSET, TEST_OFFSET = 0, 1_000_000, 2_000_000


def train_seed(i: int) -> int:
    return TRAIN_OFFSET + int(i)


def val_seed(i: int) -> int:
    return VAL_OFFSET + int(i)


def test_seed(i: int) -> int:
    return TEST_OFFSET + int(i)


def split_of(seed: int) -> str:
    return "train" if seed < VAL_OFFSET else ("val" if seed < TEST_OFFSET else "test")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_device(pref: str = "auto"):
    import torch

    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def get_logger(name: str = "uav", level: Optional[int] = None) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(level if level is not None else int(os.environ.get("UAV_LOG_LEVEL", logging.INFO)))
    return log
