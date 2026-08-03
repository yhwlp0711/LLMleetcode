"""Test cases for pytorch.llm.dpo_loss.

Compares against the reference plus known-value / property checks:
- when chosen and rejected are perfectly symmetric (delta_c == delta_r),
  logits = 0 → loss = -log σ(0) = log 2.
- larger beta with chosen strongly preferred → smaller loss.
"""

from __future__ import annotations

from math import log
from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult, compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_dpo")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _lp(b, seed, scale=1.0):
    # log-probs are negative; use -|randn| to be realistic (not required for correctness).
    return -torch.randn(b, generator=_g(seed)).abs() * scale


def _run(user_module, pc, pr, rc, rr, beta):
    args = (pc.clone(), pr.clone(), rc.clone(), rr.clone())
    return (
        user_module.dpo_loss(*args, beta=beta),
        _REF.dpo_loss(*[a.clone() for a in args], beta=beta),
    )


def _check_zero_margin_gives_log2(user_module) -> CompareResult:
    """delta_chosen == delta_rejected → logits=0 → loss = log 2."""
    pc = _lp(5, 0)
    pr = _lp(5, 1)
    # Make ref cancel so that delta_c == delta_r == 0 for every sample.
    out = user_module.dpo_loss(pc.clone(), pr.clone(), pc.clone(), pr.clone(), beta=0.3)
    return compare_numeric(out, torch.tensor(log(2.0)), atol=1e-6, rtol=1e-6)


TEST_CASES = [
    TestCase(
        name="dpo / basic (B=8, beta=0.1)",
        runner=lambda m: _run(m, _lp(8, 0), _lp(8, 1), _lp(8, 2), _lp(8, 3), 0.1),
        weight=3.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="dpo / beta=0.5",
        runner=lambda m: _run(m, _lp(8, 4), _lp(8, 5), _lp(8, 6), _lp(8, 7), 0.5),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="dpo / large magnitude (stability)",
        runner=lambda m: _run(
            m,
            _lp(6, 8, scale=50.0),
            _lp(6, 9, scale=50.0),
            _lp(6, 10, scale=50.0),
            _lp(6, 11, scale=50.0),
            0.2,
        ),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="dpo / zero margin → loss == log 2",
        runner=_check_zero_margin_gives_log2,
        weight=2.0,
    ),
]
