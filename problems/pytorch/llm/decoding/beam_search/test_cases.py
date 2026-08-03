"""Test cases for pytorch.llm.decoding.beam_search.

我们构造一个**确定性的查表式 LM**：给一个 lookup 矩阵，model_fn 返回的
logits 只依赖序列最后一个 token。这样既能复现，又足以验证解码逻辑。
"""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_beam")


VOCAB = 20
EOS = 0


def _make_lm(seed: int):
    """返回一个 model_fn(input_ids) -> logits (B, V)，行为只依赖最后一个 token。"""
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(VOCAB, VOCAB, generator=g) * 2.0  # (V, V): token -> next logits

    def model_fn(input_ids: torch.Tensor) -> torch.Tensor:
        last_tok = input_ids[:, -1]  # (B,)
        return table[last_tok]  # (B, V)

    return model_fn


def _run_beam(user_module, seed, prompt, max_len, beam_size):
    model_fn = _make_lm(seed)
    user_out = user_module.beam_search(
        model_fn, prompt.clone(), max_len=max_len, beam_size=beam_size, eos_id=EOS
    )
    ref_out = _REF.beam_search(
        model_fn, prompt.clone(), max_len=max_len, beam_size=beam_size, eos_id=EOS
    )
    return user_out, ref_out


TEST_CASES = [
    TestCase(
        name="beam / beam=2",
        runner=lambda m: _run_beam(
            m, seed=2, prompt=torch.tensor([[1, 4]]), max_len=6, beam_size=2
        ),
        weight=2.0,
    ),
    TestCase(
        name="beam / beam=3",
        runner=lambda m: _run_beam(
            m, seed=5, prompt=torch.tensor([[3, 6]]), max_len=7, beam_size=3
        ),
        weight=3.0,
    ),
    TestCase(
        name="beam / beam=4",
        runner=lambda m: _run_beam(
            m, seed=3, prompt=torch.tensor([[2, 5]]), max_len=8, beam_size=4
        ),
        weight=3.0,
    ),
]
