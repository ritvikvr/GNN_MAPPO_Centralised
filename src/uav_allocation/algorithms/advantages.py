"""Return / advantage estimators (time-step level, shared team reward)."""
from __future__ import annotations

import numpy as np


def compute_gae(rewards: np.ndarray, values: np.ndarray, last_value: float, gamma: float, lam: float):
    """Generalised Advantage Estimation for ONE episode.

    rewards[t], values[t]=V(s_t) for t=0..T-1; ``last_value`` = V(s_T) for a *truncated* episode, 0 for a *terminated* one
    (so time-limit truncation is bootstrapped instead of being treated as a true terminal state).
    Returns (advantages, returns) with returns = advantages + values.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae, next_v = 0.0, last_value
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_v = values[t]
    return adv, adv + values.astype(np.float32)


def discounted_returns(rewards: np.ndarray, last_value: float, gamma: float) -> np.ndarray:
    """Monte-Carlo discounted returns with bootstrap (used by the IPPO ablation's decentralised value)."""
    T = len(rewards)
    out = np.zeros(T, dtype=np.float32)
    g = last_value
    for t in range(T - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out
