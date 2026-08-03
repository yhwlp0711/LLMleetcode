"""Reference: Gated Activation Functions (SwiGLU / GeGLU)."""

from __future__ import annotations

from math import sqrt

import torch


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _gelu_exact(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / sqrt(2.0)))


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return _silu(gate) * x


def geglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return _gelu_exact(gate) * x
