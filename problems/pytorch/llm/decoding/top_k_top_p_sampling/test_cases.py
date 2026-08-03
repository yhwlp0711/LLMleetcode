"""Test cases for pytorch.llm.decoding.top_k_top_p_sampling."""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult, compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_topkp")


def _g(seed):
    return torch.Generator().manual_seed(seed)


def _fx(seed: int, B: int = 4, V: int = 50):
    return torch.randn(B, V, generator=_g(seed))


def _run(user_module, logits, **kwargs):
    return (
        user_module.filter_logits(logits.clone(), **kwargs),
        _REF.filter_logits(logits.clone(), **kwargs),
    )


def _check_topk_only_keeps_k(user_module) -> CompareResult:
    """Property: top-k 过滤后每行至多 k 个有限值。"""
    logits = _fx(seed=10, B=3, V=50)
    out = user_module.filter_logits(logits, top_k=5)
    finite_per_row = torch.isfinite(out).sum(dim=-1)
    if not (finite_per_row == 5).all():
        return CompareResult(
            passed=False,
            reason=f"after top_k=5, expected 5 finite per row, got {finite_per_row.tolist()}",
        )
    return CompareResult(passed=True)


def _check_topp_keeps_at_least_one(user_module) -> CompareResult:
    """Property: top-p 永远保留至少 1 个 token，即使 p 很小。"""
    logits = _fx(seed=11)
    out = user_module.filter_logits(logits, top_p=0.01)
    finite_per_row = torch.isfinite(out).sum(dim=-1)
    if (finite_per_row < 1).any():
        return CompareResult(
            passed=False,
            reason=f"top_p=0.01 should still keep ≥1 token; got {finite_per_row.tolist()}",
        )
    return CompareResult(passed=True)


TEST_CASES = [
    TestCase(
        name="temperature only / T=0.5",
        runner=lambda m: _run(m, _fx(0), temperature=0.5),
        weight=1.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="top_k only / k=5",
        runner=lambda m: _run(m, _fx(1), top_k=5),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="top_p only / p=0.9",
        runner=lambda m: _run(m, _fx(2), top_p=0.9),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="top_p only / p=0.5",
        runner=lambda m: _run(m, _fx(3), top_p=0.5),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="combined / T=0.7, k=10, p=0.95",
        runner=lambda m: _run(m, _fx(4), temperature=0.7, top_k=10, top_p=0.95),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="property / top_k=5 keeps exactly 5 per row",
        runner=_check_topk_only_keeps_k,
        weight=1.0,
    ),
    TestCase(
        name="property / top_p keeps ≥1 even at p=0.01",
        runner=_check_topp_keeps_at_least_one,
        weight=1.0,
    ),
]
