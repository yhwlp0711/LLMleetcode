"""Test cases for pytorch.basics.autograd_basics."""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(
    Path(__file__).with_name("solution.py"), "ref_autograd_basics"
)


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


# ---- grad_of_scalar: f(x) = sum(x**2); df/dx = 2*x ----


def _fx_grad_quadratic():
    return torch.randn(5, dtype=torch.float64, generator=_g(0))


def _run_grad(user_module):
    x = _fx_grad_quadratic()
    fn = lambda t: (t**2).sum()
    actual = user_module.grad_of_scalar(x.clone(), fn)
    expected = 2.0 * x
    return actual, expected


def _run_grad_complex(user_module):
    x = torch.randn(3, 4, dtype=torch.float64, generator=_g(1))
    fn = lambda t: (t.sin() * t).sum()
    # d/dx [sin(x) * x] = cos(x) * x + sin(x)
    actual = user_module.grad_of_scalar(x.clone(), fn)
    expected = x.cos() * x + x.sin()
    return actual, expected


# ---- numerical_jacobian: compare against autograd's true Jacobian ----


def _run_numjac(user_module):
    x = torch.randn(4, dtype=torch.float64, generator=_g(2))
    A = torch.randn(3, 4, dtype=torch.float64, generator=_g(3))
    b_vec = torch.randn(3, dtype=torch.float64, generator=_g(4))
    fn = lambda t: A @ t + b_vec
    actual = user_module.numerical_jacobian(fn, x.clone(), eps=1e-5)
    expected = A
    return actual, expected


def _run_numjac_nonlinear(user_module):
    x = torch.randn(3, dtype=torch.float64, generator=_g(5))
    fn = lambda t: torch.stack([t[0] ** 2, t[1] * t[2], (t[0] + t[1] + t[2]).sin()])
    # Analytical Jacobian at x:
    j_expected = torch.tensor(
        [
            [2 * x[0], 0.0, 0.0],
            [0.0, x[2], x[1]],
            [(x.sum()).cos(), (x.sum()).cos(), (x.sum()).cos()],
        ],
        dtype=torch.float64,
    )
    actual = user_module.numerical_jacobian(fn, x.clone(), eps=1e-5)
    return actual, j_expected


# ---- sgd_minimize: simple quadratic with known minimum ----


def _run_sgd_quadratic(user_module):
    # f(x) = ||x - target||^2; minimum at x = target
    target = torch.tensor([3.0, -1.0, 2.0, 0.5], dtype=torch.float64)
    fn = lambda t: ((t - target) ** 2).sum()
    x0 = torch.zeros(4, dtype=torch.float64)
    actual = user_module.sgd_minimize(fn, x0.clone(), lr=0.1, steps=100)
    expected = _REF.sgd_minimize(fn, x0.clone(), lr=0.1, steps=100)
    return actual, expected


TEST_CASES = [
    TestCase(
        name="grad_of_scalar / quadratic",
        runner=_run_grad,
        weight=1.0,
        atol=1e-8,
        rtol=1e-8,
    ),
    TestCase(
        name="grad_of_scalar / sin(x)*x",
        runner=_run_grad_complex,
        weight=1.0,
        atol=1e-8,
        rtol=1e-8,
    ),
    TestCase(
        name="numerical_jacobian / linear",
        runner=_run_numjac,
        weight=2.0,
        atol=1e-4,
        rtol=1e-4,
    ),
    TestCase(
        name="numerical_jacobian / nonlinear",
        runner=_run_numjac_nonlinear,
        weight=2.0,
        atol=1e-4,
        rtol=1e-4,
    ),
    TestCase(
        name="sgd_minimize / quadratic",
        runner=_run_sgd_quadratic,
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
]
