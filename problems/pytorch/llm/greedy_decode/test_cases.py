"""Test cases for pytorch.llm.greedy_decode.

我们构造一个**确定性的查表式 LM**：给一个 lookup 矩阵，model_fn 返回的
logits 只依赖序列最后一个 token。这样既能复现，又足以验证解码逻辑。
"""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_greedy")


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


def _run_greedy(user_module, seed, prompt, max_len):
    model_fn = _make_lm(seed)
    user_out = user_module.greedy_decode(
        model_fn, prompt.clone(), max_len=max_len, eos_id=EOS
    )
    ref_out = _REF.greedy_decode(model_fn, prompt.clone(), max_len=max_len, eos_id=EOS)
    return user_out, ref_out


TEST_CASES = [
    TestCase(
        name="greedy / short prompt, may not hit EOS",
        runner=lambda m: _run_greedy(
            m, seed=0, prompt=torch.tensor([[5, 3]]), max_len=8
        ),
        weight=2.0,
    ),
    TestCase(
        name="greedy / longer max_len",
        runner=lambda m: _run_greedy(
            m, seed=1, prompt=torch.tensor([[1, 2, 7]]), max_len=12
        ),
        weight=2.0,
    ),
    TestCase(
        name="greedy / hits EOS early",
        runner=lambda m: _run_greedy(
            m, seed=7, prompt=torch.tensor([[4, 9]]), max_len=16
        ),
        weight=2.0,
    ),
]
