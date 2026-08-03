"""Test cases for pytorch.llm.kl_penalty_estimators.

Compares each estimator against the reference, plus property checks:
- all estimators == 0 when logp == logp_ref (same distribution).
- k2 and k3 are always >= 0.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(
    Path(__file__).with_name("solution.py"), "ref_kl_estimators"
)


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _lp(shape, seed, scale=1.0):
    return torch.randn(*shape, generator=_g(seed)) * scale


def _run(user_fn, ref_fn, logp, logp_ref):
    return user_fn(logp.clone(), logp_ref.clone()), ref_fn(
        logp.clone(), logp_ref.clone()
    )


def _check_zero_when_equal(user_module) -> CompareResult:
    lp = _lp((6,), 30)
    for name, fn in [
        ("k1", user_module.kl_k1),
        ("k2", user_module.kl_k2),
        ("k3", user_module.kl_k3),
    ]:
        out = fn(lp.clone(), lp.clone())
        if out.abs().max().item() > 1e-6:
            return CompareResult(passed=False, reason=f"{name}(lp, lp) should be 0")
    return CompareResult(passed=True)


def _check_k2_k3_nonneg(user_module) -> CompareResult:
    logp = _lp((10,), 31)
    logp_ref = _lp((10,), 32)
    for name, fn in [("k2", user_module.kl_k2), ("k3", user_module.kl_k3)]:
        out = fn(logp.clone(), logp_ref.clone())
        if out.min().item() < -1e-6:
            return CompareResult(passed=False, reason=f"{name} should be >= 0")
    return CompareResult(passed=True)


TEST_CASES = [
    TestCase(
        name="k1 / matches reference",
        runner=lambda m: _run(m.kl_k1, _REF.kl_k1, _lp((8,), 0), _lp((8,), 1)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="k2 / matches reference",
        runner=lambda m: _run(m.kl_k2, _REF.kl_k2, _lp((8,), 2), _lp((8,), 3)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="k3 / matches reference",
        runner=lambda m: _run(m.kl_k3, _REF.kl_k3, _lp((8,), 4), _lp((8,), 5)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="k3 / 2D shape",
        runner=lambda m: _run(m.kl_k3, _REF.kl_k3, _lp((4, 5), 6), _lp((4, 5), 7)),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="all / == 0 when logp == logp_ref",
        runner=_check_zero_when_equal,
        weight=1.0,
    ),
    TestCase(
        name="k2,k3 / non-negativity",
        runner=_check_k2_k3_nonneg,
        weight=1.0,
    ),
]
