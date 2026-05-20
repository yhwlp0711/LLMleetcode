"""Reference solution for Tensor Ops Warmup."""
from __future__ import annotations

import torch


def flatten_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat([a.reshape(-1), b.reshape(-1)])


def row_softmax(x: torch.Tensor) -> torch.Tensor:
    m = x.max(dim=-1, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=-1, keepdim=True)


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # (a-b)^2 = a^2 - 2ab + b^2
    x2 = (x * x).sum(dim=-1, keepdim=True)               # (N, 1)
    y2 = (y * y).sum(dim=-1, keepdim=True).transpose(0, 1)  # (1, M)
    xy = x @ y.transpose(0, 1)                            # (N, M)
    return (x2 - 2 * xy + y2).clamp(min=0.0)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype).unsqueeze(-1)                    # (B, T, 1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def top_k_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    # `torch.topk` with `sorted=True` already gives descending order; ties
    # are broken by lower index in stable implementations on CPU.
    _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
    return idx
