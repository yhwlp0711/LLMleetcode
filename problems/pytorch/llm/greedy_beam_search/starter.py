"""Greedy decode 与 beam search。"""

from __future__ import annotations

from typing import Callable

import torch


def greedy_decode(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    max_len: int,
    eos_id: int,
) -> torch.Tensor:
    # TODO:
    # for _ in range(max_len):
    #     logits = model_fn(seq)               # (1, V)
    #     next_token = logits.argmax(dim=-1, keepdim=True)   # (1, 1)
    #     seq = cat([seq, next_token], dim=1)
    #     if next_token.item() == eos_id: break
    raise NotImplementedError


def beam_search(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    max_len: int,
    beam_size: int,
    eos_id: int,
) -> torch.Tensor:
    # TODO: 维护 beam_size 个候选序列 + 累积 log-prob，每步扩展再剪枝。
    # 已结束的 beam（碰到 eos）保留但不再扩展。
    # 终选时用 score = sum_log_probs / length 选最优。
    raise NotImplementedError
