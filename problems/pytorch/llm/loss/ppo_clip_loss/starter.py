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
    # TODO: 按 README 的三步实现——把 KL 惩罚并入 reward、用 GAE 估计优势、再套 PPO 裁剪目标。
    # 注意 GAE 从后往前递推，values 长度是 T+1（含 bootstrap），dones 切断跨 episode 传播。
    raise NotImplementedError
