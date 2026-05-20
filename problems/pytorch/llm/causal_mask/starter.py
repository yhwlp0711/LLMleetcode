"""因果 + Padding Mask 构造。"""

from __future__ import annotations

import torch


def build_attention_mask(pad_mask: torch.Tensor, causal: bool) -> torch.Tensor:
    # TODO: 返回 (B, 1, T, T) 布尔 mask
    # 提示：
    # 1. q_keep = pad_mask[:, :, None]   # (B, T, 1)
    # 2. k_keep = pad_mask[:, None, :]   # (B, 1, T)
    # 3. base = q_keep & k_keep          # (B, T, T)
    # 4. 如果 causal：构造下三角矩阵 tri = torch.tril(torch.ones(T, T, dtype=bool))，
    #    base = base & tri
    # 5. 在 dim=1 unsqueeze 出 head 维 → (B, 1, T, T)
    raise NotImplementedError
