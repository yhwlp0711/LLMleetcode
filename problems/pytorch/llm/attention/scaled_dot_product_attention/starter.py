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
    # TODO:
    # 1. scores = q @ k.transpose(-2, -1) / sqrt(D)
    # 2. 如果 mask 非 None：scores.masked_fill_(~mask, -inf)
    # 3. attn = softmax(scores, dim=-1)
    # 4. return attn @ v
    raise NotImplementedError
