"""Optional behaviour-cloning warm start from the decentralized local-greedy heuristic."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from ..config import Config
from ..environment.simulator import UAVTaskEnv
from ..graph.features import feature_dims
from ..graph.observations import LocalGreedyPolicy, LocalObservation
from ..models.graph_actor import GraphActor
from .rollout_buffer import RolloutBuffer


class _Recorder:
    """Wraps the expert so run-episode loops record (observation, expert action)."""
    name, centralized = "bc_expert", False

    def __init__(self, expert: LocalGreedyPolicy):
        self.expert, self.obs, self.acts = expert, [], []

    def act_index(self, o: LocalObservation) -> int:
        a = self.expert.act_index(o)
        if int(o.action_mask.sum()) > 1:
            self.obs.append(o)
            self.acts.append(a)
        return a


def collect_demos(cfg_fn: Callable[[int], Config], seeds: List[int]) -> Tuple[List[LocalObservation], List[int]]:
    from .rollout import run_episode

    obs: List[LocalObservation] = []
    acts: List[int] = []
    for s in seeds:
        cfg = cfg_fn(s)
        env = UAVTaskEnv(cfg)
        rec = _Recorder(LocalGreedyPolicy(cfg, feature_dims(cfg)))
        run_episode(env, rec, s)
        obs += rec.obs
        acts += rec.acts
    return obs, acts


def bc_train(actor: GraphActor, obs: List[LocalObservation], acts: List[int], device, epochs: int = 10, lr: float = 1e-3,
             batch_size: int = 256) -> Dict[str, float]:
    """Cross-entropy imitation of the expert over the *masked* action distribution."""
    if not obs:
        return {"bc_loss": float("nan"), "bc_acc": float("nan")}
    S = {k: torch.from_numpy(v).to(device) for k, v in RolloutBuffer.stack_obs(obs).items()}
    A = torch.as_tensor(np.asarray(acts), device=device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    n = len(acts)
    loss_v, acc = 0.0, 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            dist, _ = actor.get_action_distribution({k: v[idx] for k, v in S.items()})
            loss = -dist.log_prob(A[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            opt.step()
            tot += loss.item() * len(idx)
            correct += int((dist.probs.argmax(-1) == A[idx]).sum())
        loss_v, acc = tot / n, correct / n
    return {"bc_loss": loss_v, "bc_acc": acc, "bc_samples": float(n)}
