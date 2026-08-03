"""Reference solution for Scaled Dot-Product Attention."""
from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v
