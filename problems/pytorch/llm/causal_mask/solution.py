"""参考实现：因果 + Padding Mask 构造。"""

from __future__ import annotations

import torch


def build_attention_mask(pad_mask: torch.Tensor, causal: bool) -> torch.Tensor:
    B, T = pad_mask.shape
    q_keep = pad_mask[:, :, None]  # (B, T, 1)
    k_keep = pad_mask[:, None, :]  # (B, 1, T)
    base = q_keep & k_keep  # (B, T, T)

    if causal:
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=pad_mask.device))
        base = base & tri

    return base.unsqueeze(1)  # (B, 1, T, T)
