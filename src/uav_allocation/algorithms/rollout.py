"""Episode runners: one loop for centralized baselines, heuristics and the learned decentralized policy."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..environment.simulator import UAVTaskEnv
from ..environment.uav import Action, ActionKind, CONTINUE, RETURN, WAIT
from ..graph.features import Dims, feature_dims
from ..graph.observations import LocalObservation, collate
from ..models.graph_actor import GraphActor
from .rollout_buffer import EpisodeData


def decode_action(idx: int, o: LocalObservation, n_task: int) -> Action:
    if idx < n_task:
        return Action(ActionKind.TASK, int(o.task_ids[idx]))
    return (WAIT, RETURN, CONTINUE)[idx - n_task]


class ActorPolicy:
    """Decentralized learned policy: every UAV runs the *same shared actor* on its own local observation."""

    name = "gnn_mappo"
    centralized = False

    def __init__(self, actor: GraphActor, dims: Dims, device, deterministic: bool = False, time_single: bool = False):
        self.actor, self.dims, self.device, self.deterministic = actor, dims, device, deterministic
        self.hidden: Dict[int, torch.Tensor] = {}
        self.latencies_ms: List[float] = [] if time_single else None  # type: ignore
        self.batch_latencies_ms: List[float] = []

    def reset(self) -> None:
        self.hidden = {}

    def select(self, obs: List[LocalObservation], record: Optional[EpisodeData] = None, step: int = -1) -> List[int]:
        n_t = self.dims.n_task
        idxs = [-1] * len(obs)
        todo = [i for i, o in enumerate(obs) if int(o.action_mask.sum()) > 1]
        for i, o in enumerate(obs):
            if i not in todo:
                idxs[i] = int(np.argmax(o.action_mask))      # single legal action: no policy query, nothing to learn
        if not todo:
            return idxs
        sub = [obs[i] for i in todo]
        b = collate(sub, self.device)
        hid = None
        if self.actor.memory is not None:
            hid = torch.stack([self.hidden.get(o.ego_id, torch.zeros(self.actor.mcfg.hidden, device=self.device)) for o in sub])
        t0 = time.perf_counter()
        out = self.actor.act(b, hid, deterministic=self.deterministic)
        self.batch_latencies_ms.append((time.perf_counter() - t0) * 1e3)
        if self.latencies_ms is not None:      # per-UAV (batch size 1) inference latency, incl. collation
            b1 = collate(sub[:1], self.device)
            t1 = time.perf_counter()
            self.actor.act(b1, None if hid is None else hid[:1], deterministic=True)
            self.latencies_ms.append((time.perf_counter() - t1) * 1e3)
        acts = out["action"].cpu().numpy()
        logps = out["log_prob"].cpu().numpy()
        for k, i in enumerate(todo):
            idxs[i] = int(acts[k])
            if record is not None:
                h = hid[k].cpu().numpy() if hid is not None else None
                record.add_decision(step, sub[k], int(acts[k]), float(logps[k]), h)
            if self.actor.memory is not None:
                self.hidden[sub[k].ego_id] = out["hidden"][k].detach()
        return idxs


def run_episode(env: UAVTaskEnv, policy: Any, seed: int, options: Optional[dict] = None,
                record: Optional[EpisodeData] = None, on_step: Optional[Any] = None, frame_every: int = 1) -> Dict[str, float]:
    """Run one full episode; returns the env's summary metrics (plus wall-clock)."""
    central = bool(getattr(policy, "centralized", False))
    opts = dict(options or {})
    opts["controller"] = "centralized" if central else "decentralized"
    obs, _ = env.reset(seed=seed, options=opts)
    if hasattr(policy, "reset"):
        policy.reset()
    n_t = feature_dims(env.cfg).n_task
    done, info, t0 = False, {}, time.perf_counter()
    while not done:
        if on_step is not None and env.t % frame_every == 0:
            on_step(env)
        if central:
            actions = policy.act(env)
        else:
            ids = obs["decision_uavs"]
            actions = {}
            if ids:
                os_ = env.observe(ids)
                step_idx = -1
                if record is not None:
                    step_idx = record.add_step(env.global_state())
                if hasattr(policy, "select"):
                    idxs = policy.select(os_, record, step_idx)
                else:
                    idxs = [policy.act_index(o) for o in os_]
                actions = {o.ego_id: decode_action(i, o, n_t) for o, i in zip(os_, idxs)}
            elif record is not None:
                record.add_step(env.global_state())
        if central and record is not None:
            record.add_step(env.global_state())
        obs, r, term, trunc, info = env.step(actions)
        if record is not None:
            # a step may have been registered without decisions above; keep rewards aligned with g_states
            while len(record.rewards) < len(record.g_states) - 1:
                record.rewards.append(0.0)
            record.rewards.append(r)
        done = term or trunc
    if record is not None:
        record.terminated, record.truncated = term, trunc
        record.last_g_state = env.global_state()
    s = dict(info["episode"])
    s["wall_s"] = time.perf_counter() - t0
    return s
