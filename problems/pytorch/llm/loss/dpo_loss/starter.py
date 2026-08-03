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
    # TODO: 按 README 的 DPO 公式实现 loss（policy 相对 reference 的 log-ratio 差 → logsigmoid → 取负、平均）。
    # 数值稳定：用 logsigmoid，别写成 log(sigmoid(...))。
    raise NotImplementedError
