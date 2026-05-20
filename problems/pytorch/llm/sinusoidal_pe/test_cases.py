"""Test cases for pytorch.llm.sinusoidal_pe."""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult, compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_sinpe")


def _run(user_module, seq_len, d_model):
    return (
        user_module.build_sinusoidal_pe(seq_len, d_model),
        _REF.build_sinusoidal_pe(seq_len, d_model),
    )


def _check_position_zero(user_module) -> CompareResult:
    # 位置 0 时 sin(0)=0, cos(0)=1，所以 PE[0] 应该是 [0, 1, 0, 1, ...]
    pe = user_module.build_sinusoidal_pe(8, 16)
    expected = torch.zeros(16)
    expected[1::2] = 1.0
    return compare_numeric(pe[0], expected, atol=1e-6, rtol=1e-6)


TEST_CASES = [
    TestCase(
        name="basic / (8, 16)",
        runner=lambda m: _run(m, 8, 16),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="larger / (64, 32)",
        runner=lambda m: _run(m, 64, 32),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="odd seq_len",
        runner=lambda m: _run(m, 13, 8),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="property / position 0 = [0,1,0,1,...]",
        runner=_check_position_zero,
        weight=1.0,
    ),
]
