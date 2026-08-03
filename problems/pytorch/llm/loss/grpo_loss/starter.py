"""GRPO loss —— 组内优势归一化 + PPO 裁剪。只实现 loss 前向。"""

from __future__ import annotations

import torch


def grpo_loss(
    logratio: torch.Tensor,
    rewards: torch.Tensor,
    clip_eps: float = 0.2,
    eps_std: float = 1e-4,
) -> torch.Tensor:
    # TODO: 先把组内 rewards 做 z-score 归一化得到优势 A，再套 PPO 裁剪目标（见 README）。
    raise NotImplementedError
