"""Reference: GRPO loss (group-normalized advantage + PPO clip)."""

from __future__ import annotations

import torch


def grpo_loss(
    logratio: torch.Tensor,
    rewards: torch.Tensor,
    clip_eps: float = 0.2,
    eps_std: float = 1e-4,
) -> torch.Tensor:
    # Group-relative advantage (population std)
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps_std)

    ratio = torch.exp(logratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -torch.min(unclipped, clipped).mean()
    return loss
