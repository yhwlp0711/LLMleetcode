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
    # TODO:
    # 1. 温度缩放: logits = logits / temperature
    # 2. top-k: 找到第 k 大的阈值，小于阈值的全置 -inf
    # 3. top-p: 排序 → cumsum → 找首个超过 top_p 的位置 → 屏蔽后面
    raise NotImplementedError
