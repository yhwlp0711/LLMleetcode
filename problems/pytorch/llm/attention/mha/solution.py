"""Reference: Multi-Head Attention (Functional)."""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F


def mha(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    B, T, D = x.shape
    head_dim = D // num_heads

    q = x @ W_q
    k = x @ W_k
    v = x @ W_v

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(B, T, num_heads, head_dim).transpose(
            1, 2
        )  # (B, H, T, head_dim)

    q, k, v = split(q), split(k), split(v)

    scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ v  # (B, H, T, head_dim)

    out = out.transpose(1, 2).reshape(B, T, D)
    return out @ W_o
