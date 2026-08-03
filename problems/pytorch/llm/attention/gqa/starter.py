"""Grouped-Query Attention (GQA)。"""

from __future__ import annotations

import torch


def gqa(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # TODO: 实现 GQA。
    # 和 MHA 的唯一区别：K/V 只有 num_kv_heads 个头，需要把每个 KV 头
    # 复制给一组 Q 头共享，再做标准注意力。
    raise NotImplementedError
