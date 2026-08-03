"""PPO clipped surrogate loss —— 只实现 policy loss 前向。

不含 value loss / entropy bonus / GAE。
"""

from __future__ import annotations

import torch


def ppo_clip_loss(
    logratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    # TODO:
    #   ratio = exp(logratio)
    #   unclipped = ratio * advantages
    #   clipped   = clamp(ratio, 1-eps, 1+eps) * advantages
    #   loss = -min(unclipped, clipped).mean()
    raise NotImplementedError
