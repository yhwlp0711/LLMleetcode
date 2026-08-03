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
    # TODO: 用稳定形式
    #   loss = max(z, 0) - z * y + log(1 + exp(-|z|))
    # 再对所有元素求平均。
    raise NotImplementedError
