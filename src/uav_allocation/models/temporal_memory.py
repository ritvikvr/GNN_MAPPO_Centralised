"""Optional GRU temporal memory for the actor (one step per *decision* of a UAV)."""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMemory(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(hidden, hidden)

    def init_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = self.init_state(x.shape[0], x.device)
        return self.cell(x, h)
