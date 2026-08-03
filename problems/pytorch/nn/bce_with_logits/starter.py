"""BCE with logits —— 数值稳定的二分类交叉熵。

禁止调用 F.binary_cross_entropy_with_logits / F.binary_cross_entropy /
torch.sigmoid / F.logsigmoid。
"""

from __future__ import annotations

import torch


def bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    # TODO: 从 logits 计算数值稳定的二分类交叉熵，对所有元素求平均（见 README 的稳定公式）。
    raise NotImplementedError
