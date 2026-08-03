"""Reference: PPO loss (GAE advantage + clipped surrogate)."""

from __future__ import annotations

import torch


def ppo_clip_loss(
    logratio: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=rewards.dtype)
    gae = torch.zeros((), dtype=rewards.dtype)
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae

    ratio = torch.exp(logratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -torch.min(unclipped, clipped).mean()
    return loss
