"""参考实现：Grouped-Query Attention。"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F


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
    B, T, D = x.shape
    assert num_q_heads % num_kv_heads == 0
    repeats = num_q_heads // num_kv_heads
    head_dim = W_q.shape[1] // num_q_heads

    q = x @ W_q  # (B, T, num_q_heads * head_dim)
    k = x @ W_k  # (B, T, num_kv_heads * head_dim)
    v = x @ W_v

    # 切头：q -> (B, num_q_heads, T, head_dim)；k/v -> (B, num_kv_heads, T, head_dim)
    q = q.reshape(B, T, num_q_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, T, num_kv_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, T, num_kv_heads, head_dim).transpose(1, 2)

    # 重复 K/V head 到 num_q_heads
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)

    # SDPA
    scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ v

    # 合头 + 输出投影
    out = out.transpose(1, 2).reshape(B, T, num_q_heads * head_dim)
    return out @ W_o
