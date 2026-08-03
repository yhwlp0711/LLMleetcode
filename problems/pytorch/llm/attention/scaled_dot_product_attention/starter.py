"""Scaled Dot-Product Attention。"""

from __future__ import annotations

import torch


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 shape 为 (B, H, T_q, D_v) 的 attention 输出。

    参数:
        q: (B, H, T_q, D)
        k: (B, H, T_k, D)
        v: (B, H, T_k, D_v)
        mask: 可广播到 (B, H, T_q, T_k)；True = 保留，False = 屏蔽。
    """
    # TODO: 实现缩放点积注意力：打分（含 1/sqrt(D) 缩放）→ 应用 mask → softmax → 加权求和 v。
    raise NotImplementedError
