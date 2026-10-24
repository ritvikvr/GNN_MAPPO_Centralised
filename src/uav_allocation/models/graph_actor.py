"""Shared, edge-aware heterogeneous graph-attention actor (dense/padded implementation, no PyG dependency).

Architecture
  1. per-node-type MLP encoders (UAV / TASK / BASE) and typed edge embeddings (UU / UT / UB)
  2. L hetero message-passing layers; every relation uses an explicitly implemented GATv2-style *edge-aware*
     attention  e_ds = a^T LeakyReLU(W_l h_s + W_r h_d + W_e e_ds),  alpha = softmax_s(e), msg = sum alpha (W_l h_s + W_e e)
     relations: UAV<-UAV, UAV<-TASK, UAV<-BASE, TASK<-UAV, BASE<-UAV
  3. optional GRU memory on the ego embedding
  4. task-scoring head s_ij = MLP([h_ego, h_task_j, e_ij]) and a special-action head (WAIT / RETURN / CONTINUE)
  5. masked categorical policy over  [task slots..., WAIT, RETURN, CONTINUE]

Ablations: ``actor_type='mlp'`` (no message passing, no neighbours), ``aggregation='mean'`` (uniform instead of
attention weights), ``recurrent`` (GRU on/off).

The actor takes ONLY the local observation tensors (see graph/observations.py); it has no access to the critic or any
global simulator state.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..config import ModelConfig
from ..graph.features import Dims, N_SPECIAL
from .temporal_memory import TemporalMemory

NEG = -1e9


def mlp(i: int, h: int, o: int, norm: bool = False) -> nn.Sequential:
    layers = [nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o)]
    if norm:
        layers += [nn.LayerNorm(o)]
    return nn.Sequential(*layers)


class DenseGATv2(nn.Module):
    """Edge-aware GATv2 attention from a source node set to a destination node set (dense, masked)."""

    def __init__(self, hidden: int, heads: int, aggregation: str = "attention"):
        super().__init__()
        assert hidden % heads == 0
        self.h, self.heads, self.hd, self.agg = hidden, heads, hidden // heads, aggregation
        self.w_l = nn.Linear(hidden, hidden)
        self.w_r = nn.Linear(hidden, hidden, bias=False)
        self.w_e = nn.Linear(hidden, hidden, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.hd))
        nn.init.xavier_uniform_(self.att)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, dst: torch.Tensor, src: torch.Tensor, edge: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, Nd, _ = dst.shape
        Ns = src.shape[1]
        xl = self.w_l(src).view(B, 1, Ns, self.heads, self.hd)
        xr = self.w_r(dst).view(B, Nd, 1, self.heads, self.hd)
        ee = self.w_e(edge).view(B, Nd, Ns, self.heads, self.hd)
        m = adj.unsqueeze(-1)                                    # [B,Nd,Ns,1]
        if self.agg == "attention":
            z = F.leaky_relu(xl + xr + ee, 0.2)
            s = (z * self.att).sum(-1)                           # [B,Nd,Ns,heads]
            s = s.masked_fill(~m, NEG)
            alpha = torch.softmax(s, dim=2) * m                  # rows without neighbours -> exactly 0
        else:                                                    # mean aggregation ablation
            cnt = m.sum(dim=2, keepdim=True).clamp(min=1)
            alpha = m.float() / cnt
            alpha = alpha.expand(-1, -1, -1, self.heads)
        msg = (alpha.unsqueeze(-1) * (xl + ee)).sum(dim=2)       # [B,Nd,heads,hd]
        return self.out(msg.reshape(B, Nd, self.h))


class HeteroLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, aggregation: str):
        super().__init__()
        mk = lambda: DenseGATv2(hidden, heads, aggregation)  # noqa: E731
        self.uu, self.ut, self.ub, self.tu, self.bu = mk(), mk(), mk(), mk(), mk()
        self.norm_u, self.norm_t, self.norm_b = nn.LayerNorm(hidden), nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ff_u = mlp(hidden, hidden * 2, hidden)

    def forward(self, hu, ht, hb, eu_uu, eu_ut, eu_ub, adj_uu, adj_ut, adj_ub):
        mu = self.uu(hu, hu, eu_uu, adj_uu) + self.ut(hu, ht, eu_ut, adj_ut) + self.ub(hu, hb, eu_ub, adj_ub)
        mt = self.tu(ht, hu, eu_ut.transpose(1, 2), adj_ut.transpose(1, 2))
        mb = self.bu(hb, hu, eu_ub.transpose(1, 2), adj_ub.transpose(1, 2))
        hu = self.norm_u(hu + F.elu(mu))
        hu = hu + self.ff_u(hu)
        ht = self.norm_t(ht + F.elu(mt))
        hb = self.norm_b(hb + F.elu(mb))
        return hu, ht, hb


class MaskedCategorical(Categorical):
    """Categorical over valid actions only; invalid logits are set to a large negative number."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        self.mask = mask
        super().__init__(logits=torch.where(mask, logits, torch.full_like(logits, NEG)))


class GraphActor(nn.Module):
    def __init__(self, dims: Dims, mcfg: ModelConfig):
        super().__init__()
        self.dims, self.mcfg = dims, mcfg
        H = mcfg.hidden
        self.gnn = mcfg.actor_type == "gnn"
        self.enc_u = mlp(dims.f_uav, H, H, norm=True)
        self.enc_t = mlp(dims.f_task, H, H, norm=True)
        self.enc_b = mlp(dims.f_base, H, H, norm=True)
        self.enc_uu = mlp(dims.e_uu, H, H)
        self.enc_ut = mlp(dims.e_ut, H, H)
        self.enc_ub = mlp(dims.e_ub, H, H)
        self.layers = nn.ModuleList([HeteroLayer(H, mcfg.heads, mcfg.aggregation) for _ in range(mcfg.layers)]) if self.gnn else nn.ModuleList()
        self.memory = TemporalMemory(H) if mcfg.recurrent else None
        self.score = nn.Sequential(nn.Linear(2 * H + dims.e_ut, H), nn.ReLU(), nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
        self.special = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, N_SPECIAL))
        self.value_head = mlp(H, H, 1)     # used only by the IPPO ablation (decentralised value); never by act()

    # ---------------------------------------------------------------- 1. encode
    def encode_graph(self, b: Dict[str, torch.Tensor], hidden: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hu, ht, hb = self.enc_u(b["uav_x"]), self.enc_t(b["task_x"]), self.enc_b(b["base_x"])
        if self.gnn:
            e_uu, e_ut, e_ub = self.enc_uu(b["e_uu"]), self.enc_ut(b["e_ut"]), self.enc_ub(b["e_ub"])
            for layer in self.layers:
                hu, ht, hb = layer(hu, ht, hb, e_uu, e_ut, e_ub, b["adj_uu"], b["adj_ut"], b["adj_ub"])
        h_ego = hu[:, 0]
        new_hidden = None
        if self.memory is not None:
            h_ego = self.memory(h_ego, hidden)
            new_hidden = h_ego
        hb_pool = (hb * b["base_mask"].unsqueeze(-1)).sum(1) / b["base_mask"].sum(1, keepdim=True).clamp(min=1)
        return {"h_ego": h_ego, "h_task": ht, "h_base": hb_pool, "hidden": new_hidden}

    # ---------------------------------------------------------------- 2. score
    def score_candidates(self, enc: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Logits over [task slots..., WAIT, RETURN, CONTINUE] (unmasked)."""
        B, Nt, _ = enc["h_task"].shape
        ego = enc["h_ego"].unsqueeze(1).expand(-1, Nt, -1)
        s = self.score(torch.cat([ego, enc["h_task"], b["e_ut"][:, 0]], dim=-1)).squeeze(-1)      # s_ij = MLP([h_i, h_j, e_ij])
        sp = self.special(torch.cat([enc["h_ego"], enc["h_base"]], dim=-1))
        return torch.cat([s, sp], dim=-1)

    # ---------------------------------------------------------------- 3. distribution
    def get_action_distribution(self, b: Dict[str, torch.Tensor], hidden: Optional[torch.Tensor] = None
                                ) -> Tuple[MaskedCategorical, Dict[str, torch.Tensor]]:
        enc = self.encode_graph(b, hidden)
        logits = self.score_candidates(enc, b)
        mask = b["action_mask"].clone()
        empty = ~mask.any(dim=-1)
        if empty.any():                                   # defensive: hold is always a legal fallback
            mask[empty, self.dims.n_task] = True
        return MaskedCategorical(logits, mask), enc

    # ---------------------------------------------------------------- 4. act
    @torch.no_grad()
    def act(self, b: Dict[str, torch.Tensor], hidden: Optional[torch.Tensor] = None, deterministic: bool = False
            ) -> Dict[str, torch.Tensor]:
        dist, enc = self.get_action_distribution(b, hidden)
        a = dist.probs.argmax(-1) if deterministic else dist.sample()
        return {"action": a, "log_prob": dist.log_prob(a), "entropy": dist.entropy(), "hidden": enc["hidden"]}

    def local_value(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value_head(enc["h_ego"]).squeeze(-1)
