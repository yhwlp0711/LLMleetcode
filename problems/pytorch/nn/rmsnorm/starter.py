"""RMSNorm 模块。"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_dim: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        # TODO: self.weight = nn.Parameter(torch.ones(normalized_dim))
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps); return x / rms * weight
        raise NotImplementedError
