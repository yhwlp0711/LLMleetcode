"""LayerNorm 模块 —— 实现一个 nn.Module 子类。"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_dim: int, eps: float = 1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
