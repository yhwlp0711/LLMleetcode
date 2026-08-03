"""Reference: BCE with logits (numerically stable)."""

from __future__ import annotations

import torch


def bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    z = logits
    # Stable: max(z, 0) - z * y + log(1 + exp(-|z|))
    loss = z.clamp(min=0) - z * target + torch.log1p(torch.exp(-z.abs()))
    return loss.mean()
