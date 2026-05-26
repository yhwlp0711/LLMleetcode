"""Reference: Activation Functions."""

from __future__ import annotations

from math import pi, sqrt

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / sqrt(2.0)))


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    c = sqrt(2.0 / pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return silu(gate) * x


def geglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return gelu_exact(gate) * x


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
