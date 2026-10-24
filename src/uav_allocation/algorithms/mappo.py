"""MAPPO (shared actor + centralized critic) and its IPPO ablation, with checkpointing.

Objective (per decision i at step t, shared team advantage A_t):
    L_pi = -E[ min(r A, clip(r, 1-eps, 1+eps) A) ] - c_ent * H[pi]        r = pi_new(a|o) / pi_old(a|o)
    L_V  = E[(V(s_t) - R_t)^2]                                             (centralized critic on ground-truth state)
A_t, R_t come from GAE on the time-step level with the single shared reward r_t. Per-agent log-probabilities are used
(each UAV is a PPO agent sharing parameters), which is the standard MAPPO factorisation.
IPPO ablation: no centralized critic; each decision is baselined by the actor's own *local* value head V_i(o_i) and
Monte-Carlo discounted returns (the decentralised critic cannot be trained on time-level GAE, since agents decide asynchronously).
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from ..config import Config, config_from_dict
from ..graph.features import Dims, Norm, feature_dims
from ..graph.observations import collate_global
from ..models.centralized_critic import CentralizedCritic
from ..models.graph_actor import GraphActor
from .advantages import compute_gae, discounted_returns
from .rollout_buffer import RolloutBuffer


class MAPPO:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg, self.tc, self.device = cfg, cfg.train, device
        self.dims = feature_dims(cfg)
        self.actor = GraphActor(self.dims, cfg.model).to(device)
        self.ippo = cfg.train.algo == "ippo"
        self.critic = None if self.ippo else CentralizedCritic(cfg).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.tc.lr_actor, eps=1e-5)
        self.critic_opt = None if self.ippo else torch.optim.Adam(self.critic.parameters(), lr=self.tc.lr_critic, eps=1e-5)
        self.iteration = 0
        self.best_metric = -float("inf")
        self.sched: List[Any] = []
        if self.tc.lr_schedule:
            f = lambda it: max(0.1, 1.0 - it / max(self.tc.iterations, 1))  # noqa: E731
            self.sched.append(torch.optim.lr_scheduler.LambdaLR(self.actor_opt, f))
            if self.critic_opt is not None:
                self.sched.append(torch.optim.lr_scheduler.LambdaLR(self.critic_opt, f))

    # ------------------------------------------------------------------ advantage computation
    @torch.no_grad()
    def _values(self, buf: RolloutBuffer) -> List[Tuple[np.ndarray, float]]:
        out = []
        for ep in buf.episodes:
            if self.critic is None:
                out.append((np.zeros(len(ep.rewards), dtype=np.float32), 0.0))
                continue
            vs = []
            for i in range(0, len(ep.g_states), 256):
                vs.append(self.critic(collate_global(ep.g_states[i:i + 256], self.device)).cpu().numpy())
            v = np.concatenate(vs) if vs else np.zeros(0, dtype=np.float32)
            last = 0.0
            if ep.truncated and ep.last_g_state is not None:           # bootstrap time-limit truncation
                last = float(self.critic(collate_global([ep.last_g_state], self.device)).item())
            out.append((v.astype(np.float32), last))
        return out

    # ------------------------------------------------------------------ update
    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        tc, dev = self.tc, self.device
        t0 = time.perf_counter()
        vals = self._values(buf)
        g_states, returns_t, adv_t, offsets, off = [], [], [], [], 0
        obs, d_glob_step, acts, old_logp, hid = [], [], [], [], []
        mc_returns: List[float] = []
        for ep, (v, last) in zip(buf.episodes, vals):
            r = np.asarray(ep.rewards, dtype=np.float32)
            if self.critic is not None:
                adv, ret = compute_gae(r, v, last, tc.gamma, tc.gae_lambda)
                adv_t.append(adv)
                returns_t.append(ret)
                g_states += ep.g_states
            else:
                mc = discounted_returns(r, 0.0, tc.gamma)
                mc_returns.extend(mc[s] for s in ep.d_step)
            obs += ep.obs
            d_glob_step += [off + s for s in ep.d_step]
            acts += ep.actions
            old_logp += ep.logps
            hid += ep.hiddens
            off += len(ep.rewards)
        n = len(acts)
        stats: Dict[str, float] = {"n_decisions": float(n), "n_steps": float(off)}
        if n == 0:
            return stats
        S = RolloutBuffer.stack_obs(obs)
        S = {k: torch.from_numpy(v).to(dev) for k, v in S.items()}
        A = torch.as_tensor(np.asarray(acts), device=dev)
        OLP = torch.as_tensor(np.asarray(old_logp, dtype=np.float32), device=dev)
        H = None
        if self.actor.memory is not None:
            H = torch.as_tensor(np.stack(hid), device=dev, dtype=torch.float32)
        if self.critic is not None:
            adv_all = torch.as_tensor(np.concatenate(adv_t), device=dev)
            ret_all = torch.as_tensor(np.concatenate(returns_t), device=dev)
            step_idx = torch.as_tensor(np.asarray(d_glob_step), device=dev)
            ADV = adv_all[step_idx]
            G = collate_global(g_states, None)
        else:
            MC = torch.as_tensor(np.asarray(mc_returns, dtype=np.float32), device=dev)
        agg = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0}
        n_mb = 0
        for _ in range(tc.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, tc.minibatch_size):
                idx = perm[i:i + tc.minibatch_size]
                b = {k: v[idx] for k, v in S.items()}
                dist, enc = self.actor.get_action_distribution(b, None if H is None else H[idx])
                logp = dist.log_prob(A[idx])
                ent = dist.entropy().mean()
                ratio = torch.exp(logp - OLP[idx])
                if self.critic is not None:
                    adv = ADV[idx]
                else:
                    adv = MC[idx] - self.actor.local_value(enc).detach()
                if tc.normalize_adv and len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - tc.clip, 1 + tc.clip) * adv
                pi_loss = -torch.min(s1, s2).mean()
                loss = pi_loss - tc.ent_coef * ent
                v_loss = torch.zeros((), device=dev)
                if self.critic is None:
                    v_loss = ((self.actor.local_value(enc) - MC[idx]) ** 2).mean()
                    loss = loss + tc.vf_coef * v_loss
                if self.critic is None or self.iteration >= tc.critic_warmup_iters:
                    self.actor_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), tc.max_grad_norm)
                    self.actor_opt.step()
                agg["pi_loss"] += pi_loss.item()
                agg["entropy"] += ent.item()
                agg["kl"] += (OLP[idx] - logp).mean().item()
                agg["clip_frac"] += ((ratio - 1).abs() > tc.clip).float().mean().item()
                agg["v_loss"] += v_loss.item()
                n_mb += 1
            if self.critic is not None:                      # centralized critic: minibatches of time steps
                T = ret_all.shape[0]
                perm_t = torch.randperm(T)
                for i in range(0, T, tc.minibatch_size):
                    idt = perm_t[i:i + tc.minibatch_size]
                    g = {k: v[idt].to(dev) for k, v in G.items()}
                    vloss = ((self.critic(g) - ret_all[idt.to(dev)]) ** 2).mean()
                    self.critic_opt.zero_grad()
                    (tc.vf_coef * vloss).backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), tc.max_grad_norm)
                    self.critic_opt.step()
                    agg["v_loss"] += vloss.item()
        for k, v in agg.items():
            stats[k] = v / max(n_mb, 1)
        if self.critic is not None:
            stats["v_loss"] = agg["v_loss"] / max(tc.epochs * int(np.ceil(ret_all.shape[0] / tc.minibatch_size)), 1)
        for s in self.sched:
            s.step()
        stats["update_s"] = time.perf_counter() - t0
        return stats

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ck = {
            "actor": self.actor.state_dict(),
            "critic": None if self.critic is None else self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": None if self.critic_opt is None else self.critic_opt.state_dict(),
            "config": self.cfg.to_dict(),
            "norm": asdict(Norm.from_cfg(self.cfg)),
            "iteration": self.iteration,
            "best_metric": self.best_metric,
            "extra": extra or {},
        }
        torch.save(ck, path)

    @classmethod
    def load(cls, path: str, device: torch.device, cfg: Optional[Config] = None) -> "MAPPO":
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = cfg or config_from_dict(ck["config"])
        m = cls(cfg, device)
        m.actor.load_state_dict(ck["actor"])
        if m.critic is not None and ck.get("critic") is not None:
            m.critic.load_state_dict(ck["critic"])
        m.actor_opt.load_state_dict(ck["actor_opt"])
        if m.critic_opt is not None and ck.get("critic_opt") is not None:
            m.critic_opt.load_state_dict(ck["critic_opt"])
        m.iteration = int(ck.get("iteration", 0))
        m.best_metric = float(ck.get("best_metric", -float("inf")))
        return m


def load_actor(path: str, device: torch.device) -> Tuple[GraphActor, Config, Dims]:
    """Load ONLY the actor (decentralized deployment: no critic is constructed or loaded)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(ck["config"])
    dims = feature_dims(cfg)
    actor = GraphActor(dims, cfg.model).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor, cfg, dims
