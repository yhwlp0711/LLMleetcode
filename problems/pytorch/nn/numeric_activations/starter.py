"""数值稳定的 Sigmoid / Softmax —— 按数学定义实现。

禁止调用 torch.sigmoid / F.softmax。
"""

from __future__ import annotations

import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    # TODO: 数值稳定的 sigmoid（对 x≥0 / x<0 分别处理，避免 exp 溢出）。
    raise NotImplementedError


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # TODO: 数值稳定的 softmax（沿 dim）。
    raise NotImplementedError
