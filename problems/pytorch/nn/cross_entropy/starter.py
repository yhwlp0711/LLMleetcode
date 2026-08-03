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
    # TODO:
    #   1. log_softmax(logits)  —— 数值稳定（先减 max，或用 logsumexp）
    #   2. 取出每个样本真实类别的 log-prob
    #   3. 忽略 target == ignore_index 的样本
    #   4. 对有效样本取负、求平均
    raise NotImplementedError
