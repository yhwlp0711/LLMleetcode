"""数值稳定的 Sigmoid / Softmax —— 按数学定义实现。

禁止调用 torch.sigmoid / F.softmax。
"""

from __future__ import annotations

import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    # TODO: 1 / (1 + exp(-x))，要求数值稳定（对 x>=0 / x<0 分别处理）
    raise NotImplementedError


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # TODO: 数值稳定的 softmax（先减 max 再 exp）
    raise NotImplementedError
