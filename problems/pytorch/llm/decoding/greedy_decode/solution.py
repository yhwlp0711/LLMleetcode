"""参考实现：Greedy decode。"""

from __future__ import annotations

from typing import Callable

import torch


def greedy_decode(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    max_len: int,
    eos_id: int,
) -> torch.Tensor:
    seq = input_ids.clone()
    for _ in range(max_len):
        logits = model_fn(seq)  # (1, V)
        next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
        seq = torch.cat([seq, next_token], dim=1)
        if int(next_token.item()) == eos_id:
            break
    return seq
