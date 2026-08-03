"""Test cases for pytorch.llm.attention.causal_mask."""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_causal_mask")


def _run(user_module, pad_mask, causal):
    return (
        user_module.build_attention_mask(pad_mask.clone(), causal),
        _REF.build_attention_mask(pad_mask.clone(), causal),
    )


def _all_real():
    return torch.ones(2, 6, dtype=torch.bool)


def _with_padding():
    # First sequence: 4 real + 2 pad; Second: 6 real.
    return torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )


def _all_padded_row():
    # Edge: one sequence completely padded (rare in practice but worth testing)
    return torch.tensor(
        [
            [True, True, True, False, False, False],
            [False, False, False, False, False, False],
        ]
    )


TEST_CASES = [
    TestCase(
        name="no padding / no causal",
        runner=lambda m: _run(m, _all_real(), False),
        weight=1.0,
    ),
    TestCase(
        name="no padding / causal",
        runner=lambda m: _run(m, _all_real(), True),
        weight=1.0,
    ),
    TestCase(
        name="with padding / no causal",
        runner=lambda m: _run(m, _with_padding(), False),
        weight=2.0,
    ),
    TestCase(
        name="with padding / causal",
        runner=lambda m: _run(m, _with_padding(), True),
        weight=2.0,
    ),
    TestCase(
        name="edge / fully padded row",
        runner=lambda m: _run(m, _all_padded_row(), True),
        weight=1.0,
    ),
]
