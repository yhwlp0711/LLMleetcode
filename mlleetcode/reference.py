"""Reference implementations exposed to problem solvers.

Some integration problems (e.g. transformer_block) need building blocks that
were the subject of earlier problems. To avoid coupling the integration test
to the user's own past implementations (which may be buggy), we expose
verified reference versions here for them to import.
"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional RMSNorm: x / RMS(x) * weight, eps inside sqrt."""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention. mask True = keep."""
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v
