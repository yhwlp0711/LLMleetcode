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


# ---- Single-input activations ---------------------------------------------


def _check_single(user_fn, ref_fn, x):
    return user_fn(x.clone()), ref_fn(x.clone())


# ---- Gated activations ----------------------------------------------------


def _check_gated(user_fn, ref_fn, x, gate):
    return user_fn(x.clone(), gate.clone()), ref_fn(x.clone(), gate.clone())


# ---- Cross-checks against PyTorch's built-ins (extra safety net) ----------


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
    # --- SwiGLU ---
    TestCase(
        name="swiglu / order check (SiLU on `gate`, not `x`)",
        runner=lambda m: _check_gated(
            m.swiglu, _REF.swiglu, _typical((4, 8), 6), _typical((4, 8), 7)
        ),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="swiglu / 3D shape (B, T, D)",
        runner=lambda m: _check_gated(
            m.swiglu, _REF.swiglu, _typical((2, 5, 8), 8), _typical((2, 5, 8), 9)
        ),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    # --- GeGLU ---
    TestCase(
        name="geglu / order check",
        runner=lambda m: _check_gated(
            m.geglu, _REF.geglu, _typical((4, 8), 10), _typical((4, 8), 11)
        ),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    # --- Sigmoid ---
    TestCase(
        name="sigmoid / typical values",
        runner=lambda m: _check_single(m.sigmoid, _REF.sigmoid, _typical((4, 8), 20)),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="sigmoid / extreme values (stability)",
        runner=lambda m: _check_single(m.sigmoid, _REF.sigmoid, _extreme()),
        weight=1.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="sigmoid / matches torch.sigmoid",
        runner=lambda m: _check_against_torch_builtin(
            m.sigmoid, torch.sigmoid, _typical((6, 8), 21)
        ),
        weight=1.0,
    ),
    # --- Softmax ---
    TestCase(
        name="softmax / dim=-1",
        runner=lambda m: (
            m.softmax(_typical((4, 8), 30), dim=-1),
            _REF.softmax(_typical((4, 8), 30), dim=-1),
        ),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="softmax / dim=0",
        runner=lambda m: (
            m.softmax(_typical((4, 8), 31), dim=0),
            _REF.softmax(_typical((4, 8), 31), dim=0),
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
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="softmax / matches F.softmax",
        runner=lambda m: _check_against_torch_builtin(
            lambda t: m.softmax(t, dim=-1),
            lambda t: F.softmax(t, dim=-1),
            _typical((6, 8), 32),
        ),
        weight=1.0,
    ),
]
