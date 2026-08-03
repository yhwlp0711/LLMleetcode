"""因果 + Padding Mask 构造。"""

from __future__ import annotations

import torch


def build_attention_mask(pad_mask: torch.Tensor, causal: bool) -> torch.Tensor:
    # TODO: 返回 (B, 1, T, T) 布尔 mask（True=保留）。
    # padding 位置两两屏蔽；causal=True 时再叠加下三角约束（只能看当前及之前）。
    raise NotImplementedError
