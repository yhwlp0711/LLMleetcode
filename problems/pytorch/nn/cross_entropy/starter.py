"""交叉熵损失 —— 从 logits 直接计算，数值稳定，支持 ignore_index。

禁止调用 F.cross_entropy / F.nll_loss / F.log_softmax。
"""

from __future__ import annotations

import torch


def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    # TODO: 从 logits 计算交叉熵（数值稳定的 log-softmax），跳过 target==ignore_index 的样本，对有效样本求平均。
    raise NotImplementedError
