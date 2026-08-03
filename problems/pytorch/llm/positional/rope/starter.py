"""Rotary Position Embeddings (RoPE)。"""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: 构造 (seq_len, head_dim) 的 cos / sin 表。
    # 成对复制：cos[m, 2i] == cos[m, 2i+1] == cos(m * theta_i)。
    raise NotImplementedError


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # TODO: 把 RoPE 应用到 shape 为 (B, H, T, head_dim) 的 x 上。
    # 技巧：构造 x_rotated，每对 (a, b) 变成 (-b, a)，
    #       然后 out = x * cos + x_rotated * sin。
    raise NotImplementedError
