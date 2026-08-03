"""KL 散度 —— 从 logits 计算 forward KL(P ‖ Q)，数值稳定。

禁止调用 F.kl_div / F.log_softmax / F.softmax。
"""

from __future__ import annotations

import torch


def kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    # TODO:
    #   log_p = p_logits - logsumexp(p_logits)     # 稳定 log-softmax
    #   log_q = q_logits - logsumexp(q_logits)
    #   p = exp(log_p)
    #   kl_per_sample = sum_c p * (log_p - log_q)   # 沿类别维求和
    #   return kl_per_sample.mean()                 # batch 平均
    raise NotImplementedError
