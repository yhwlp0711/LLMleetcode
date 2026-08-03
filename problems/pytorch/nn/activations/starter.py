"""激活函数 —— 按数学定义实现。

禁止调用 F.gelu / F.silu / nn.GELU / nn.SiLU。
"""

from __future__ import annotations

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    # TODO: x * sigmoid(x)
    raise NotImplementedError


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    # TODO: 0.5 * x * (1 + erf(x / sqrt(2)))
    raise NotImplementedError


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    # TODO: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    raise NotImplementedError
