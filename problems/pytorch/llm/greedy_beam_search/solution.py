"""参考实现：Greedy decode 与 beam search。"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


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


def beam_search(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    max_len: int,
    beam_size: int,
    eos_id: int,
) -> torch.Tensor:
    # 初始化：用 prompt 跑一次拿 first-step logits
    logits = model_fn(input_ids)  # (1, V)
    log_probs = F.log_softmax(logits[0], dim=-1)  # (V,)
    topv, topi = log_probs.topk(beam_size)  # (K,) (K,)

    beams = torch.cat(
        [
            input_ids.expand(beam_size, -1),
            topi.unsqueeze(1),
        ],
        dim=1,
    )  # (K, T_init + 1)
    scores = topv  # (K,) 累积 log-prob
    finished = torch.zeros(beam_size, dtype=torch.bool)  # 标记 beam 是否已结束
    lengths = torch.ones(beam_size, dtype=torch.long)  # 生成的 token 数

    # 标记初始 beam 中已经是 eos 的
    finished |= topi == eos_id

    for step in range(1, max_len):
        if finished.all():
            break

        next_logits = model_fn(beams)  # (K, V)
        log_p = F.log_softmax(next_logits, dim=-1)  # (K, V)
        V = log_p.shape[-1]

        # 对已结束的 beam，强制只能继续生成 eos（不改变 score、length 也不增加）
        # 做法：把 finished beam 的 next token 概率改成「eos=0, others=-inf」
        # 这样 expand 时这些 beam 的 score 不变
        masked_log_p = log_p.clone()
        if finished.any():
            masked_log_p[finished] = float("-inf")
            masked_log_p[finished, eos_id] = 0.0

        # 累加成 (K, V) 候选 score
        cand_scores = scores.unsqueeze(1) + masked_log_p  # (K, V)
        flat = cand_scores.view(-1)  # (K * V,)
        topv, topi = flat.topk(beam_size)  # (K,) (K,)

        beam_idx = topi // V
        token_idx = topi % V

        beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = topv

        # 更新 finished / lengths：之前已 finished 的不变；新生成的 token 若是 eos 也算 finished
        was_finished = finished[beam_idx]
        new_finished = (token_idx == eos_id) & ~was_finished
        finished = was_finished | new_finished
        lengths = lengths[beam_idx] + (~was_finished).long()

    # 长度归一化挑选
    norm_scores = scores / lengths.float().clamp(min=1)
    best = norm_scores.argmax()
    return beams[best : best + 1]
