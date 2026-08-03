"""Reference: Activation Functions (SiLU / GELU)."""

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
