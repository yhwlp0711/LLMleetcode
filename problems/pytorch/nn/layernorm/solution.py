"""Reference: LayerNorm Module."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_dim: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_dim))
        self.bias = nn.Parameter(torch.zeros(normalized_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias
