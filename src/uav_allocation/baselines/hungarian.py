"""Baseline C (centralized): Hungarian one-to-one assignment (scipy.optimize.linear_sum_assignment).

Cost = -utility for feasible pairs (shifted to be >= 0). Infeasible pairs get a *bounded* prohibitive cost
``BIG = (min(nu, nt) + 1) * C_max + 1`` where ``C_max`` is the largest finite cost. Because BIG exceeds the total cost of
any feasible matching, the optimum first maximises the number of feasible pairs and then minimises cost; any
assignment that lands on a BIG entry is discarded (tasks / UAVs that cannot be matched stay unassigned).

NOTE: this solves a single-shot linear assignment (one task per UAV per call). It does NOT solve the multi-task routing /
scheduling problem (no sequences, time windows, or future arrivals) - it is re-run every step on the idle UAVs.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..environment.uav import Action
from .common import build_candidates, to_actions


class HungarianAssignment:
    name = "hungarian"
    centralized = True

    def act(self, env) -> Dict[int, Action]:
        c = build_candidates(env)
        if not c.uav_ids or not c.task_ids or not c.feasible.any():
            return {}
        cost = np.where(c.feasible, c.utility.max() - c.utility, np.nan)
        cmax = float(np.nanmax(cost)) if np.isfinite(cost).any() else 1.0
        big = (min(cost.shape) + 1) * max(cmax, 1e-6) + 1.0
        cost = np.where(c.feasible, cost, big)
        rows, cols = linear_sum_assignment(cost)
        pairs = [(c.uav_ids[i], c.task_ids[j]) for i, j in zip(rows, cols) if c.feasible[i, j]]
        return to_actions(pairs)
