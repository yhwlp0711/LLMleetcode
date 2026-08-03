"""GRPO loss —— 组内优势归一化 + PPO 裁剪。只实现 loss 前向。"""

from __future__ import annotations

import torch


def grpo_loss(
    logratio: torch.Tensor,
    rewards: torch.Tensor,
    clip_eps: float = 0.2,
    eps_std: float = 1e-4,
) -> torch.Tensor:
    # TODO:
    #   1. 组内优势: A = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps_std)
    #   2. PPO 裁剪:
    #        ratio = exp(logratio)
    #        loss = -min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A).mean()
    raise NotImplementedError
