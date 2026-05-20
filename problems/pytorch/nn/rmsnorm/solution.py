"""Reference: RMSNorm Module (LLaMA-style)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_dim: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
