"""Test cases for pytorch.llm.ppo_clip_loss.

Compares against the reference plus a known-value check:
- when logratio == 0 (ratio == 1, i.e. π_new == π_old), loss = -mean(advantages).
"""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_ppo")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _randn(b, seed, scale=1.0):
    return torch.randn(b, generator=_g(seed)) * scale


def _run(user_module, logratio, adv, eps):
    return (
        user_module.ppo_clip_loss(logratio.clone(), adv.clone(), eps),
        _REF.ppo_clip_loss(logratio.clone(), adv.clone(), eps),
    )


def _check_zero_logratio(user_module):
    """ratio == 1 → loss = -mean(advantages)."""
    adv = _randn(8, 20)
    lr = torch.zeros(8)
    out = user_module.ppo_clip_loss(lr.clone(), adv.clone(), 0.2)
    return compare_numeric(out, -adv.mean(), atol=1e-6, rtol=1e-6)


TEST_CASES = [
    TestCase(
        name="ppo / small logratio (unclipped region)",
        runner=lambda m: _run(m, _randn(8, 0, scale=0.05), _randn(8, 1), 0.2),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ppo / large logratio triggers clip (eps=0.2)",
        runner=lambda m: _run(m, _randn(10, 2, scale=1.0), _randn(10, 3), 0.2),
        weight=3.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ppo / eps=0.1, mixed-sign advantages",
        runner=lambda m: _run(m, _randn(10, 4, scale=0.8), _randn(10, 5), 0.1),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ppo / logratio==0 → loss = -mean(A)",
        runner=_check_zero_logratio,
        weight=2.0,
    ),
]
