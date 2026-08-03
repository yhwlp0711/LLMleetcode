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
    # TODO:
    # 1. q = x @ W_q          # (B, T, num_q_heads * head_dim)
    #    k = x @ W_k          # (B, T, num_kv_heads * head_dim)
    #    v = x @ W_v
    # 2. reshape + transpose 切头 → q: (B, num_q_heads, T, head_dim)
    #                              k/v: (B, num_kv_heads, T, head_dim)
    # 3. repeat_interleave(k, repeats=num_q_heads // num_kv_heads, dim=1)
    #    同理 v；现在 k/v 也是 num_q_heads 个头
    # 4. SDPA：scores → softmax → @ v
    # 5. transpose + reshape 合头 → @ W_o
    raise NotImplementedError
