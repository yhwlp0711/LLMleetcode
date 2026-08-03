"""带 KV Cache 的 SDPA。"""

from __future__ import annotations

import torch


def sdpa_with_kv_cache(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO:
    # 1. 拼接 cache：new_k_cache = cat([k_cache, k_new], dim=-2) （cache 为 None 时直接用 k_new）
    # 2. 同理拼 v
    # 3. SDPA: scores = q_new @ new_k.transpose(-2,-1) / sqrt(D)
    #          attn = softmax(scores, dim=-1); out = attn @ new_v
    # 4. return out, new_k_cache, new_v_cache
    raise NotImplementedError
