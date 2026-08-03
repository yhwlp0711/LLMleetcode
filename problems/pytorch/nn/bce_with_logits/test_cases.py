"""Test cases for pytorch.nn.bce_with_logits.

Compares against the reference and against F.binary_cross_entropy_with_logits,
including extreme logits to catch numerical instability.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_bce")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _logits(shape, seed, scale=1.0):
    return torch.randn(*shape, generator=_g(seed)) * scale


def _target(shape, seed):
    return (torch.rand(*shape, generator=_g(seed)) > 0.5).float()


def _run(user_module, logits, target):
    return (
        user_module.bce_with_logits(logits.clone(), target.clone()),
        _REF.bce_with_logits(logits.clone(), target.clone()),
    )


def _run_vs_torch(user_module, logits, target):
    actual = user_module.bce_with_logits(logits.clone(), target.clone())
    expected = F.binary_cross_entropy_with_logits(
        logits.clone(), target.clone(), reduction="mean"
    )
    return compare_numeric(actual, expected, atol=1e-6, rtol=1e-6)


TEST_CASES = [
    TestCase(
        name="bce / basic (N=8)",
        runner=lambda m: _run(m, _logits((8,), 0), _target((8,), 1)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="bce / 2D multi-label (4, 5)",
        runner=lambda m: _run(m, _logits((4, 5), 2), _target((4, 5), 3)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="bce / matches F.bce_with_logits",
        runner=lambda m: _run_vs_torch(m, _logits((10,), 4), _target((10,), 5)),
        weight=2.0,
    ),
    TestCase(
        name="bce / extreme logits (stability)",
        runner=lambda m: _run(m, _logits((6,), 6, scale=300.0), _target((6,), 7)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
]
