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
    # TODO: 把新的 k/v 拼接到历史 cache 后面，再用 q_new 对完整 k/v 做 SDPA。
    # 返回 (输出, 更新后的 k_cache, 更新后的 v_cache)；cache 为 None 表示首步。
    raise NotImplementedError
