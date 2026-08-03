"""Greedy decode。"""

from __future__ import annotations

from typing import Callable

import torch


def greedy_decode(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    max_len: int,
    eos_id: int,
) -> torch.Tensor:
    # TODO: 每步取 argmax 作为下一个 token，追加到序列；遇到 eos 或达到 max_len 停止。
    raise NotImplementedError
