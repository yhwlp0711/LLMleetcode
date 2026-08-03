"""PPO loss —— KL 惩罚(k3) + GAE 优势估计 + 裁剪代理损失。只实现 loss 前向。

不含训练循环 / 反向 / value loss / entropy bonus。
"""

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
    # TODO:
    # 步骤 1 —— KL 惩罚并入 reward（k3 估计器，logr = logp_ref - logp）:
    #   kl = exp(logr) - 1 - logr
    #   r' = rewards - kl_coef * kl
    # 步骤 2 —— GAE（用 r'，从后往前）:
    #   delta_t = r'[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
    #   adv[t]  = delta_t + gamma * lam * (1 - dones[t]) * adv[t+1]   # adv[T]=0
    # 步骤 3 —— 裁剪损失:
    #   ratio = exp(logratio)
    #   loss  = -min(ratio * adv, clamp(ratio, 1-eps, 1+eps) * adv).mean()
    raise NotImplementedError
