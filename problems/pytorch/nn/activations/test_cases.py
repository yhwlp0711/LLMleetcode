"""Test cases for pytorch.nn.activations.

Compares each activation against the reference implementation on several
input shapes / value ranges, including extreme magnitudes (to catch silent
overflow / underflow).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_activations")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _typical(shape, seed: int):
    return torch.randn(*shape, generator=_g(seed))


def _extreme():
    # Mix of large positive / large negative / zero / typical values to test
    # numerical behaviour.
    return torch.tensor([-50.0, -5.0, -1.0, 0.0, 1.0, 5.0, 50.0])


def _check_single(user_fn, ref_fn, x):
    return user_fn(x.clone()), ref_fn(x.clone())


def _check_against_torch_builtin(user_fn, builtin, x):
    """Verify the user's implementation also agrees with F.silu / F.gelu."""
    return compare_numeric(user_fn(x.clone()), builtin(x.clone()), atol=1e-6, rtol=1e-6)


TEST_CASES = [
    # --- SiLU ---
    TestCase(
        name="silu / typical values",
        runner=lambda m: _check_single(m.silu, _REF.silu, _typical((4, 8), 0)),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="silu / extreme values",
        runner=lambda m: _check_single(m.silu, _REF.silu, _extreme()),
        weight=1.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="silu / matches F.silu",
        runner=lambda m: _check_against_torch_builtin(
            m.silu, F.silu, _typical((6, 8), 1)
        ),
        weight=1.0,
    ),
    # --- GELU exact ---
    TestCase(
        name="gelu_exact / typical",
        runner=lambda m: _check_single(
            m.gelu_exact, _REF.gelu_exact, _typical((4, 8), 2)
        ),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="gelu_exact / matches F.gelu",
        runner=lambda m: _check_against_torch_builtin(
            m.gelu_exact,
            lambda t: F.gelu(t, approximate="none"),
            _typical((6, 8), 3),
        ),
        weight=1.0,
    ),
    # --- GELU tanh approximation ---
    TestCase(
        name="gelu_tanh / typical",
        runner=lambda m: _check_single(
            m.gelu_tanh, _REF.gelu_tanh, _typical((4, 8), 4)
        ),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="gelu_tanh / matches F.gelu(approximate='tanh')",
        runner=lambda m: _check_against_torch_builtin(
            m.gelu_tanh,
            lambda t: F.gelu(t, approximate="tanh"),
            _typical((6, 8), 5),
        ),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
]
