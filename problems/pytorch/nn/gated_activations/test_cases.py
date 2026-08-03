"""Test cases for pytorch.nn.gated_activations.

Compares swiglu / geglu against the reference on several shapes, plus a gate
ordering check (SiLU/GELU must be applied to `gate`, not `x`).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(
    Path(__file__).with_name("solution.py"), "ref_gated_activations"
)


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _typical(shape, seed: int):
    return torch.randn(*shape, generator=_g(seed))


def _check_gated(user_fn, ref_fn, x, gate):
    return user_fn(x.clone(), gate.clone()), ref_fn(x.clone(), gate.clone())


def _check_swiglu_order(user_module):
    """SwiGLU must apply SiLU to `gate`, not `x`. Use asymmetric inputs so a
    swapped implementation gives a different result."""
    x = _typical((4, 8), 40)
    gate = _typical((4, 8), 41)
    return user_module.swiglu(x.clone(), gate.clone()), F.silu(gate.clone()) * x.clone()


def _check_geglu_order(user_module):
    """GeGLU must apply GELU to `gate`, not `x`."""
    x = _typical((4, 8), 42)
    gate = _typical((4, 8), 43)
    ref = F.gelu(gate.clone(), approximate="none") * x.clone()
    return user_module.geglu(x.clone(), gate.clone()), ref


TEST_CASES = [
    # --- SwiGLU ---
    TestCase(
        name="swiglu / typical",
        runner=lambda m: _check_gated(
            m.swiglu, _REF.swiglu, _typical((4, 8), 2), _typical((4, 8), 3)
        ),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="swiglu / order check (SiLU on `gate`, not `x`)",
        runner=_check_swiglu_order,
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="swiglu / 3D shape (B, T, D)",
        runner=lambda m: _check_gated(
            m.swiglu, _REF.swiglu, _typical((2, 5, 8), 4), _typical((2, 5, 8), 5)
        ),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    # --- GeGLU ---
    TestCase(
        name="geglu / typical",
        runner=lambda m: _check_gated(
            m.geglu, _REF.geglu, _typical((4, 8), 6), _typical((4, 8), 7)
        ),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="geglu / order check (GELU on `gate`, not `x`)",
        runner=_check_geglu_order,
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="geglu / 3D shape (B, T, D)",
        runner=lambda m: _check_gated(
            m.geglu, _REF.geglu, _typical((2, 5, 8), 8), _typical((2, 5, 8), 9)
        ),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
]
