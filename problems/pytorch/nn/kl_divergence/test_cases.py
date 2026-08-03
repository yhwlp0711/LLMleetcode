"""Test cases for pytorch.nn.kl_divergence.

Compares against the reference and against F.kl_div (batchmean, which computes
Σ p·(log p − log q) averaged over the batch — matching our forward-KL / mean
convention). Also checks KL(P‖P) == 0 and large-logit stability.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult, compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_kl")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _logits(n, c, seed, scale=1.0):
    return torch.randn(n, c, generator=_g(seed)) * scale


def _run(user_module, p_logits, q_logits):
    return (
        user_module.kl_divergence(p_logits.clone(), q_logits.clone()),
        _REF.kl_divergence(p_logits.clone(), q_logits.clone()),
    )


def _run_vs_torch(user_module, p_logits, q_logits):
    actual = user_module.kl_divergence(p_logits.clone(), q_logits.clone())
    log_p = F.log_softmax(p_logits.clone(), dim=-1)
    log_q = F.log_softmax(q_logits.clone(), dim=-1)
    # F.kl_div(input=log_q, target=p) = Σ p·(log p − log q); batchmean divides by N.
    expected = F.kl_div(log_q, log_p.exp(), reduction="batchmean")
    return compare_numeric(actual, expected, atol=1e-6, rtol=1e-6)


def _check_self_kl_zero(user_module):
    """KL(P ‖ P) must be 0."""
    p = _logits(6, 8, 30)
    out = user_module.kl_divergence(p.clone(), p.clone())
    return compare_numeric(out, torch.tensor(0.0), atol=1e-6, rtol=1e-6)


def _check_nonnegative(user_module) -> CompareResult:
    """KL ≥ 0 always."""
    val = float(user_module.kl_divergence(_logits(8, 10, 31), _logits(8, 10, 32)))
    if val < -1e-6:
        return CompareResult(passed=False, reason=f"KL should be ≥ 0, got {val}")
    return CompareResult(passed=True)


TEST_CASES = [
    TestCase(
        name="kl / basic (N=8, C=6)",
        runner=lambda m: _run(m, _logits(8, 6, 0), _logits(8, 6, 1)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="kl / matches F.kl_div (batchmean)",
        runner=lambda m: _run_vs_torch(m, _logits(10, 7, 2), _logits(10, 7, 3)),
        weight=2.0,
    ),
    TestCase(
        name="kl / large logits (stability)",
        runner=lambda m: _run(
            m, _logits(6, 5, 4, scale=300.0), _logits(6, 5, 5, scale=300.0)
        ),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="kl / KL(P‖P) == 0",
        runner=_check_self_kl_zero,
        weight=1.0,
    ),
    TestCase(
        name="kl / non-negativity",
        runner=_check_nonnegative,
        weight=1.0,
    ),
]
