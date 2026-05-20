"""参考实现：带 KV Cache 的 SDPA。"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F


def sdpa_with_kv_cache(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 拼接 K/V 沿 T 维（dim=-2）
    if k_cache is not None and k_cache.numel() > 0:
        new_k_cache = torch.cat([k_cache, k_new], dim=-2)
        new_v_cache = torch.cat([v_cache, v_new], dim=-2)
    else:
        new_k_cache = k_new
        new_v_cache = v_new

    # 标准 SDPA
    d = q_new.shape[-1]
    scores = q_new @ new_k_cache.transpose(-2, -1) / sqrt(d)
    attn = F.softmax(scores, dim=-1)
    out = attn @ new_v_cache

    return out, new_k_cache, new_v_cache
