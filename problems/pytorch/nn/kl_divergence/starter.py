"""KL 散度 —— 从 logits 计算 forward KL(P ‖ Q)，数值稳定。

禁止调用 F.kl_div / F.log_softmax / F.softmax。
"""

from __future__ import annotations

import torch


def kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    # TODO: 从两组 logits 计算 forward KL(P‖Q)，逐样本求和后对 batch 平均（在 log 空间计算以保证稳定）。
    raise NotImplementedError
