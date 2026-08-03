"""DPO 损失 —— Direct Preference Optimization 的 loss 前向计算。

只实现 loss，不涉及训练循环 / log-prob 的计算。
"""

from __future__ import annotations

import torch


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    # TODO:
    #   delta_chosen   = policy_chosen_logps   - ref_chosen_logps
    #   delta_rejected = policy_rejected_logps - ref_rejected_logps
    #   logits = beta * (delta_chosen - delta_rejected)
    #   loss = -logsigmoid(logits)            # 用 logsigmoid，别用 log(sigmoid(...))
    #   return loss.mean()
    raise NotImplementedError
