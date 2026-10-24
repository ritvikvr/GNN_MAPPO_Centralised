"""Baseline A (centralized): nearest feasible UAV.

Tasks are processed in order of (priority desc, deadline asc); each is given to the still-free feasible UAV with the
smallest nominal travel time.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..environment.uav import Action
from .common import build_candidates, to_actions


class NearestFeasible:
    name = "nearest"
    centralized = True

    def act(self, env) -> Dict[int, Action]:
        c = build_candidates(env)
        if not c.uav_ids or not c.task_ids:
            return {}
        order = sorted(range(len(c.task_ids)), key=lambda j: (-env.tasks[c.task_ids[j]].priority, env.tasks[c.task_ids[j]].deadline))
        free = np.ones(len(c.uav_ids), dtype=bool)
        pairs = []
        for j in order:
            ok = free & c.feasible[:, j]
            if not ok.any():
                continue
            i = int(np.argmin(np.where(ok, c.eta[:, j], np.inf)))
            free[i] = False
            pairs.append((c.uav_ids[i], c.task_ids[j]))
        return to_actions(pairs)
