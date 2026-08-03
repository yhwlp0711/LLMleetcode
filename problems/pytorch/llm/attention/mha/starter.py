"""Multi-Head Attention（纯函数版）。"""

from __future__ import annotations

import torch


def mha(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # TODO: 投影 Q/K/V → 切分多头 → SDPA → 合并多头 → 输出投影
    raise NotImplementedError
