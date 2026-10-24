"""Baseline B (centralized): greedy utility auction.

Repeatedly awards the (UAV, task) pair with the highest weighted utility
(priority, travel time, energy, deadline urgency, workload) until no feasible pair is left.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..environment.uav import Action
from .common import build_candidates, to_actions


class GreedyAuction:
    name = "greedy"
    centralized = True

    def act(self, env) -> Dict[int, Action]:
        c = build_candidates(env)
        if not c.uav_ids or not c.task_ids:
            return {}
        score = np.where(c.feasible, c.utility, -np.inf)
        pairs = []
        while np.isfinite(score).any():
            i, j = np.unravel_index(int(np.argmax(score)), score.shape)
            pairs.append((c.uav_ids[i], c.task_ids[j]))
            score[i, :] = -np.inf
            score[:, j] = -np.inf
        return to_actions(pairs)
