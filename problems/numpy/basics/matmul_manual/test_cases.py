"""Test cases for numpy.basics.matmul_manual."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(
    Path(__file__).with_name("solution.py"), "ref_matmul_manual"
)


def _rng(seed: int):
    return np.random.default_rng(seed)


def _fx_matmul_small():
    return _rng(0).standard_normal((4, 3)), _rng(1).standard_normal((3, 5))


def _fx_matmul_rect():
    return _rng(2).standard_normal((10, 16)), _rng(3).standard_normal((16, 7))


def _fx_transpose():
    return _rng(4).standard_normal((6, 9))


def _fx_batch():
    return _rng(5).standard_normal((4, 3, 5)), _rng(6).standard_normal((4, 5, 2))


TEST_CASES = [
    TestCase(
        name="matmul / small square-ish",
        runner=lambda m: (
            m.matmul(*_fx_matmul_small()),
            _REF.matmul(*_fx_matmul_small()),
        ),
        weight=2.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="matmul / wider K",
        runner=lambda m: (
            m.matmul(*_fx_matmul_rect()),
            _REF.matmul(*_fx_matmul_rect()),
        ),
        weight=2.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="transpose_2d",
        runner=lambda m: (
            m.transpose_2d(_fx_transpose()),
            _REF.transpose_2d(_fx_transpose()),
        ),
        weight=1.0,
        atol=1e-12,
        rtol=1e-12,
    ),
    TestCase(
        name="batched_matmul",
        runner=lambda m: (
            m.batched_matmul(*_fx_batch()),
            _REF.batched_matmul(*_fx_batch()),
        ),
        weight=2.0,
        atol=1e-10,
        rtol=1e-10,
    ),
]
