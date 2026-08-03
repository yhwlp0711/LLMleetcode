"""Test cases for pytorch.nn.numeric_activations.

Checks sigmoid / softmax against the reference and PyTorch built-ins, including
extreme magnitudes to catch silent overflow / underflow.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult, compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(
    Path(__file__).with_name("solution.py"), "ref_numeric_activations"
)


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _typical(shape, seed: int):
    return torch.randn(*shape, generator=_g(seed))


def _extreme():
    # Include magnitudes large enough to overflow a naive 1/(1+exp(-x)) in fp32
    # (exp(1000) = inf), so unstable implementations produce nan/inf and fail.
    return torch.tensor([-1000.0, -100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0, 1000.0])


def _check_single(user_fn, ref_fn, x):
    return user_fn(x.clone()), ref_fn(x.clone())


def _check_against_torch_builtin(user_fn, builtin, x):
    return compare_numeric(user_fn(x.clone()), builtin(x.clone()), atol=1e-6, rtol=1e-6)


def _check_sigmoid_finite(user_module) -> CompareResult:
    """A stable sigmoid must stay finite even for very large-magnitude inputs;
    a naive 1/(1+exp(-x)) overflows to nan/inf here."""
    x = torch.tensor([-1000.0, -300.0, 0.0, 300.0, 1000.0])
    out = user_module.sigmoid(x.clone())
    if not torch.isfinite(out).all():
        return CompareResult(
            passed=False, reason="sigmoid produced nan/inf on large inputs"
        )
    return CompareResult(passed=True)


TEST_CASES = [
    # --- Sigmoid ---
    TestCase(
        name="sigmoid / typical values",
        runner=lambda m: _check_single(m.sigmoid, _REF.sigmoid, _typical((4, 8), 0)),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="sigmoid / extreme values (stability)",
        runner=lambda m: _check_single(m.sigmoid, _REF.sigmoid, _extreme()),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="sigmoid / stays finite on huge inputs",
        runner=_check_sigmoid_finite,
        weight=2.0,
    ),
    TestCase(
        name="sigmoid / matches torch.sigmoid",
        runner=lambda m: _check_against_torch_builtin(
            m.sigmoid, torch.sigmoid, _typical((6, 8), 1)
        ),
        weight=1.0,
    ),
    # --- Softmax ---
    TestCase(
        name="softmax / dim=-1",
        runner=lambda m: (
            m.softmax(_typical((4, 8), 2), dim=-1),
            _REF.softmax(_typical((4, 8), 2), dim=-1),
        ),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="softmax / dim=0",
        runner=lambda m: (
            m.softmax(_typical((4, 8), 3), dim=0),
            _REF.softmax(_typical((4, 8), 3), dim=0),
        ),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="softmax / extreme values (stability)",
        runner=lambda m: (
            m.softmax(torch.tensor([[1000.0, 1.0, -1000.0]]), dim=-1),
            _REF.softmax(torch.tensor([[1000.0, 1.0, -1000.0]]), dim=-1),
        ),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="softmax / matches F.softmax",
        runner=lambda m: _check_against_torch_builtin(
            lambda t: m.softmax(t, dim=-1),
            lambda t: F.softmax(t, dim=-1),
            _typical((6, 8), 4),
        ),
        weight=1.0,
    ),
]
