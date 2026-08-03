"""PPO loss —— GAE 优势估计 + 裁剪代理损失。只实现 loss 前向。

不含训练循环 / 反向 / value loss / entropy bonus。
"""

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
    # TODO:
    # 步骤 1 —— GAE（从后往前）:
    #   delta_t = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
    #   adv[t]  = delta_t + gamma * lam * (1 - dones[t]) * adv[t+1]   # adv[T]=0
    # 步骤 2 —— 裁剪损失:
    #   ratio = exp(logratio)
    #   loss  = -min(ratio * adv, clamp(ratio, 1-eps, 1+eps) * adv).mean()
    raise NotImplementedError
