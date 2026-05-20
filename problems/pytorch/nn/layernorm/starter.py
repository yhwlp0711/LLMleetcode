"""LayerNorm 模块 —— 实现一个 nn.Module 子类。"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_dim: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        # TODO: 创建 self.weight 和 self.bias 作为 nn.Parameter
        # self.weight = nn.Parameter(torch.ones(normalized_dim))
        # self.bias = nn.Parameter(torch.zeros(normalized_dim))
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 沿最后一维归一化，使用有偏方差
        raise NotImplementedError
