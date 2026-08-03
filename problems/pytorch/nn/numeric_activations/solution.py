"""Reference: numerically-stable Sigmoid / Softmax."""

from __future__ import annotations

import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    # Numerically stable: avoid overflow of exp(-x) for very negative x
    pos = x >= 0
    neg = ~pos
    out = torch.empty_like(x)
    out[pos] = 1.0 / (1.0 + torch.exp(-x[pos]))
    exp_x = torch.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=dim, keepdim=True)
