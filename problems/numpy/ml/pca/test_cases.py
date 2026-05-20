"""Test cases for numpy.ml.pca."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_pca")


def _rng(seed: int):
    return np.random.default_rng(seed)


def _make_data(seed: int, N: int, D: int):
    return _rng(seed).standard_normal((N, D))


def _run(user_module, X, k, idx):
    user_out = user_module.pca(X.copy(), n_components=k)
    ref_out = _REF.pca(X.copy(), n_components=k)
    return user_out[idx], ref_out[idx]


def _check_reconstruction(user_module) -> CompareResult:
    """Property: 投影 + 反投影 ≈ 原数据（取所有主成分时严格相等）。"""
    X = _make_data(seed=99, N=30, D=5)
    components, _, projected = user_module.pca(X.copy(), n_components=5)  # 取全部 5 个
    X_centered = X - X.mean(axis=0, keepdims=True)
    reconstructed = projected @ components
    diff = np.abs(reconstructed - X_centered).max()
    if diff > 1e-8:
        return CompareResult(
            passed=False,
            reason=f"reconstruction max diff {diff:.2e} (expected < 1e-8)",
        )
    return CompareResult(passed=True)


_X1 = _make_data(seed=0, N=50, D=8)
_X2 = _make_data(seed=1, N=100, D=20)


TEST_CASES = [
    TestCase(
        name="small / components",
        runner=lambda m: _run(m, _X1, 3, 0),
        weight=2.0,
        atol=1e-8,
        rtol=1e-8,
    ),
    TestCase(
        name="small / explained_var",
        runner=lambda m: _run(m, _X1, 3, 1),
        weight=1.0,
        atol=1e-8,
        rtol=1e-8,
    ),
    TestCase(
        name="small / projected",
        runner=lambda m: _run(m, _X1, 3, 2),
        weight=2.0,
        atol=1e-8,
        rtol=1e-8,
    ),
    TestCase(
        name="medium / components (k=5, D=20)",
        runner=lambda m: _run(m, _X2, 5, 0),
        weight=2.0,
        atol=1e-8,
        rtol=1e-8,
    ),
    TestCase(
        name="property / full-rank reconstruction",
        runner=_check_reconstruction,
        weight=1.0,
    ),
]
