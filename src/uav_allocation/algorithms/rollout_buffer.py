"""Rollout storage for multi-agent PPO with a variable number of decisions per time step.

Per time step we store the privileged global state and the team reward; per *decision* (one UAV choosing an action at a
step) we store the local observation, action, behaviour log-prob and (if recurrent) the incoming hidden state.
Episodes are kept separate so GAE never crosses episode boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..graph.observations import LOCAL_KEYS, LocalObservation


@dataclass
class EpisodeData:
    g_states: List[Dict[str, np.ndarray]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    obs: List[LocalObservation] = field(default_factory=list)
    d_step: List[int] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    hiddens: List[Optional[np.ndarray]] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    last_g_state: Optional[Dict[str, np.ndarray]] = None

    def add_step(self, g_state: Dict[str, np.ndarray]) -> int:
        self.g_states.append(g_state)
        return len(self.g_states) - 1

    def add_decision(self, step: int, o: LocalObservation, a: int, logp: float, hidden: Optional[np.ndarray]) -> None:
        self.obs.append(o)
        self.d_step.append(step)
        self.actions.append(a)
        self.logps.append(logp)
        self.hiddens.append(hidden)


class RolloutBuffer:
    def __init__(self) -> None:
        self.episodes: List[EpisodeData] = []

    def add(self, ep: EpisodeData) -> None:
        self.episodes.append(ep)

    def __len__(self) -> int:
        return sum(len(e.actions) for e in self.episodes)

    def clear(self) -> None:
        self.episodes = []

    def n_steps(self) -> int:
        return sum(len(e.rewards) for e in self.episodes)

    @staticmethod
    def stack_obs(obs: List[LocalObservation]) -> Dict[str, np.ndarray]:
        return {k: np.stack([getattr(o, k) for o in obs]) for k in LOCAL_KEYS}
