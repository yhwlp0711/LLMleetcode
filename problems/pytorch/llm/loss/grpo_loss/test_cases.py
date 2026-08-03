"""Test cases for pytorch.llm.loss.grpo_loss.

Compares against the reference plus property checks:
- advantage is group-normalized: mean(A) ≈ 0.
- when logratio == 0 (ratio == 1), loss = -mean(A) ≈ 0.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_grpo")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _randn(n, seed, scale=1.0):
    return torch.randn(n, generator=_g(seed)) * scale


def _run(user_module, logratio, rewards, eps):
    return (
        user_module.grpo_loss(logratio.clone(), rewards.clone(), eps),
        _REF.grpo_loss(logratio.clone(), rewards.clone(), eps),
    )


def _check_zero_logratio_gives_zero(user_module):
    """ratio == 1 → loss = -mean(A); A is group-normalized so mean(A) ≈ 0 → loss ≈ 0."""
    rewards = _randn(8, 20)
    lr = torch.zeros(8)
    out = user_module.grpo_loss(lr.clone(), rewards.clone(), 0.2)
    return compare_numeric(out, torch.tensor(0.0), atol=1e-4, rtol=1e-4)


TEST_CASES = [
    TestCase(
        name="grpo / small logratio, G=6",
        runner=lambda m: _run(m, _randn(6, 0, scale=0.05), _randn(6, 1), 0.2),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="grpo / large logratio triggers clip, G=8",
        runner=lambda m: _run(m, _randn(8, 2, scale=1.0), _randn(8, 3), 0.2),
        weight=3.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="grpo / eps=0.1, G=10",
        runner=lambda m: _run(m, _randn(10, 4, scale=0.8), _randn(10, 5), 0.1),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="grpo / logratio==0 → loss ≈ 0 (group-normalized)",
        runner=_check_zero_logratio_gives_zero,
        weight=2.0,
    ),
]
