"""Centralized training critic (CTDE): fully-connected global graph attention over ALL UAV and task nodes using
privileged ground-truth state. Never imported by the actor and never used at execution time."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import Config
from ..graph.features import global_dims
from .graph_actor import mlp


class CentralizedCritic(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        H, heads, L = cfg.model.hidden, cfg.model.heads, cfg.model.layers
        fu, ft, fs = global_dims(cfg)
        self.enc_u, self.enc_t = mlp(fu, H, H, norm=True), mlp(ft, H, H, norm=True)
        self.type_emb = nn.Parameter(torch.zeros(2, H))
        layer = nn.TransformerEncoderLayer(H, heads, dim_feedforward=2 * H, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=L, enable_nested_tensor=False)
        self.enc_s = mlp(fs, H, H)
        self.head = nn.Sequential(nn.Linear(3 * H, H), nn.ReLU(), nn.Linear(H, 1))

    def forward(self, g: Dict[str, torch.Tensor]) -> torch.Tensor:
        um, tm = g["g_uav_mask"], g["g_task_mask"]
        # trim padding to the largest populated slot in this batch (big speed-up; masks keep results identical)
        nu_ = int(um.any(0).nonzero().max()) + 1
        nt_ = int(tm.any(0).nonzero().max()) + 1 if tm.any() else 1
        g = dict(g, g_uav=g["g_uav"][:, :nu_], g_task=g["g_task"][:, :nt_])
        um, tm = um[:, :nu_], tm[:, :nt_]
        hu = self.enc_u(g["g_uav"]) + self.type_emb[0]
        ht = self.enc_t(g["g_task"]) + self.type_emb[1]
        x = torch.cat([hu, ht], dim=1)
        valid = torch.cat([um, tm], dim=1)
        x = self.encoder(x, src_key_padding_mask=~valid)
        nu = hu.shape[1]
        xu, xt = x[:, :nu], x[:, nu:]
        pu = (xu * um.unsqueeze(-1)).sum(1) / um.sum(1, keepdim=True).clamp(min=1)
        pt = (xt * tm.unsqueeze(-1)).sum(1) / tm.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([pu, pt, self.enc_s(g["g_scalar"])], dim=-1)).squeeze(-1)
