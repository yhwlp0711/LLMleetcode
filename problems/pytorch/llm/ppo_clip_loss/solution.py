"""Reference: PPO clipped surrogate loss."""

from __future__ import annotations

import torch


def ppo_clip_loss(
    logratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(logratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(unclipped, clipped).mean()
    return loss
