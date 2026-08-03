"""参考实现：Top-k / Top-p Logits 过滤。"""

from __future__ import annotations

import torch


def filter_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    out = logits / temperature

    # Top-k filtering
    if top_k > 0:
        k = min(top_k, out.shape[-1])
        # threshold = 第 k 大的值（每行各自的阈值）
        kth = out.topk(k, dim=-1).values[..., -1, None]  # (B, 1)
        out = torch.where(out < kth, torch.full_like(out, float("-inf")), out)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_idx = out.sort(dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = probs.cumsum(dim=-1)

        # 标记要移除的位置：累积概率 > top_p 之后的全部 token
        sorted_remove = cum_probs > top_p
        # 右移一位：累积值刚刚超过 top_p 的 token 也要保留
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False  # 至少保留 1 个

        # 把排序后的 mask 散回原索引顺序
        remove_mask = torch.zeros_like(sorted_remove)
        remove_mask.scatter_(dim=-1, index=sorted_idx, src=sorted_remove)
        out = out.masked_fill(remove_mask, float("-inf"))

    return out
