"""Test cases for pytorch.nn.cross_entropy.

Compares against the reference and against F.cross_entropy (with the same
ignore_index / mean reduction) as an extra safety net, including large-magnitude
logits to catch numerical instability.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_ce")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _logits(n, c, seed, scale=1.0):
    return torch.randn(n, c, generator=_g(seed)) * scale


def _target(n, c, seed):
    return torch.randint(0, c, (n,), generator=_g(seed))


def _run(user_module, logits, target, ignore_index=-100):
    return (
        user_module.cross_entropy(logits.clone(), target.clone(), ignore_index),
        _REF.cross_entropy(logits.clone(), target.clone(), ignore_index),
    )


def _run_vs_torch(user_module, logits, target, ignore_index=-100):
    actual = user_module.cross_entropy(logits.clone(), target.clone(), ignore_index)
    expected = F.cross_entropy(
        logits.clone(), target.clone(), ignore_index=ignore_index, reduction="mean"
    )
    return compare_numeric(actual, expected, atol=1e-6, rtol=1e-6)


def _target_with_ignore(n, c, seed, n_ignore):
    t = _target(n, c, seed)
    idx = torch.arange(n)[:n_ignore]
    t[idx] = -100
    return t


TEST_CASES = [
    TestCase(
        name="ce / basic (N=8, C=5)",
        runner=lambda m: _run(m, _logits(8, 5, 0), _target(8, 5, 1)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ce / matches F.cross_entropy",
        runner=lambda m: _run_vs_torch(m, _logits(10, 7, 2), _target(10, 7, 3)),
        weight=2.0,
    ),
    TestCase(
        name="ce / large logits (stability)",
        runner=lambda m: _run(m, _logits(6, 4, 4, scale=100.0), _target(6, 4, 5)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ce / with ignore_index",
        runner=lambda m: _run(m, _logits(12, 6, 6), _target_with_ignore(12, 6, 7, 4)),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ce / ignore_index matches F.cross_entropy",
        runner=lambda m: _run_vs_torch(
            m, _logits(12, 6, 8), _target_with_ignore(12, 6, 9, 5)
        ),
        weight=2.0,
    ),
]
