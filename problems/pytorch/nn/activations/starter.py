"""激活函数 —— 按数学定义实现。

禁止调用 F.gelu / F.silu / F.glu / F.softmax / torch.sigmoid / nn.GELU / nn.SiLU。
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


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # TODO: SiLU(gate) * x  （注意顺序：SiLU 作用在 gate 上）
    raise NotImplementedError


def geglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # TODO: GELU_exact(gate) * x
    raise NotImplementedError


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    # TODO: 1 / (1 + exp(-x))，要求数值稳定
    raise NotImplementedError


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # TODO: 数值稳定的 softmax（先减 max 再 exp）
    raise NotImplementedError
