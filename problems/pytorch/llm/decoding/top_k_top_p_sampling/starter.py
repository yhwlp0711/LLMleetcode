"""Top-k / Top-p Sampling — 实现 logits 过滤。"""

from __future__ import annotations

import torch


def filter_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    # TODO: 依次做温度缩放、top-k 过滤、top-p（nucleus）过滤，被过滤的 logits 置 -inf。
    raise NotImplementedError
