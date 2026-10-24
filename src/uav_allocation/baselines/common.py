"""Shared helpers for the centralized baselines.

All three baselines are CENTRALIZED: they read the simulator's ground truth (every UAV's state and every task) and
command UAVs over the ground link (which suffers outages and ``comm.command_loss``). They use exactly the same task
definitions, energy model and hard feasibility checker as the learned method.
They only assign *idle* UAVs to *pending, un-owned* tasks (no pre-emption / re-routing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..coordination.bidding import compute_bid
from ..environment.task import TaskInfo, TaskStatus
from ..environment.uav import Action, ActionKind, UAVStatus
from ..safety.feasibility import TaskArrays


@dataclass
class Candidates:
    uav_ids: List[int]
    task_ids: List[int]
    feasible: np.ndarray    # [nu, nt] bool
    eta: np.ndarray         # [nu, nt]
    utility: np.ndarray     # [nu, nt] (only meaningful where feasible)


def build_candidates(env) -> Candidates:
    t = env.t
    uids = [u.id for u in env.uavs if u.status == UAVStatus.IDLE and u.comm_ok(t) and not env.checker.must_return(u)]
    tasks = [x for x in env.tasks if x.announced and x.status == TaskStatus.PENDING]
    tids = [x.id for x in tasks]
    nu, nt = len(uids), len(tids)
    feas = np.zeros((nu, nt), dtype=bool)
    eta = np.zeros((nu, nt))
    util = np.zeros((nu, nt))
    if nu and nt:
        arr = TaskArrays.from_infos([TaskInfo.from_task(x) for x in tasks])
        for i, uid in enumerate(uids):
            u = env.uavs[uid]
            f = env.checker.check(u, arr, t)
            feas[i], eta[i] = f.mask, f.eta
            slack = arr.deadline - (t + f.eta + arr.service)
            for j in np.nonzero(f.mask)[0]:
                util[i, j] = compute_bid(arr.priority[j], f.eta[j], arr.service[j], slack[j], f.e_travel[j], u.capacity)
            util[i] -= 0.002 * u.tasks_done      # mild workload balancing term
    return Candidates(uids, tids, feas, eta, util)


def to_actions(pairs: List[tuple[int, int]]) -> Dict[int, Action]:
    return {u: Action(ActionKind.TASK, t) for u, t in pairs}
