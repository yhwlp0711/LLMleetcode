"""Reference: PPO loss (KL penalty k3 + GAE advantage + clipped surrogate)."""

from __future__ import annotations

import torch


def ppo_clip_loss(
    logratio: torch.Tensor,
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    kl_coef: float = 0.1,
) -> torch.Tensor:
    # Step 1: KL penalty (k3) folded into reward
    logr = logp_ref - logp
    kl = torch.exp(logr) - 1.0 - logr
    r = rewards - kl_coef * kl

    # Step 2: GAE (backward), using penalized reward r
    T = r.shape[0]
    adv = torch.zeros(T, dtype=r.dtype)
    gae = torch.zeros((), dtype=r.dtype)
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = r[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae

    # Step 3: clipped surrogate
    ratio = torch.exp(logratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -torch.min(unclipped, clipped).mean()
    return loss
