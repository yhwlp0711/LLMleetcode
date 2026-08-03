"""Rotary Position Embeddings (RoPE)。"""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: 预计算 (seq_len, head_dim) 的 cos / sin 查找表。
    # 每对相邻特征共用一个频率（cos[m, 2i] == cos[m, 2i+1]）。
    raise NotImplementedError


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # TODO: 把 RoPE 应用到 (B, H, T, head_dim) 的 x 上：对每对相邻特征按位置角度做二维旋转。
    raise NotImplementedError
