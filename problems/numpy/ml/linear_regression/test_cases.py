"""Test cases for 001 Linear Regression.

We use the reference solution to compute expected outputs on the fly, so the
problem author only needs to define the input fixtures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_001_solution")


def _make_dataset(seed: int, n_train: int, n_test: int, d: int):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, d))
    true_w = rng.standard_normal(d)
    true_b = rng.standard_normal()
    noise = 0.1 * rng.standard_normal(n_train)
    y_train = X_train @ true_w + true_b + noise
    X_test = rng.standard_normal((n_test, d))
    return X_train, y_train, X_test


def _run(user_module, fixture, lr, epochs, output_idx):
    X_train, y_train, X_test = fixture
    actual = user_module.fit_predict(
        X_train.copy(), y_train.copy(), X_test.copy(),
        lr=lr, epochs=epochs,
    )
    expected = _REF.fit_predict(
        X_train.copy(), y_train.copy(), X_test.copy(),
        lr=lr, epochs=epochs,
    )
    return actual[output_idx], expected[output_idx]


# Three fixtures: small/medium/larger
_F1 = _make_dataset(seed=0, n_train=50, n_test=10, d=3)
_F2 = _make_dataset(seed=1, n_train=200, n_test=40, d=5)
_F3 = _make_dataset(seed=2, n_train=500, n_test=100, d=8)


TEST_CASES = [
    TestCase(
        name="small / weights w",
        runner=lambda m: _run(m, _F1, lr=0.05, epochs=200, output_idx=0),
        weight=1.0, atol=1e-6, rtol=1e-6,
        description="3 features, 200 epochs — compare learned weights w.",
    ),
    TestCase(
        name="small / bias b",
        runner=lambda m: _run(m, _F1, lr=0.05, epochs=200, output_idx=1),
        weight=1.0, atol=1e-6, rtol=1e-6,
    ),
    TestCase(
        name="medium / predictions on X_test",
        runner=lambda m: _run(m, _F2, lr=0.01, epochs=500, output_idx=2),
        weight=2.0, atol=1e-6, rtol=1e-6,
        description="5 features, 500 epochs — compare test predictions.",
    ),
    TestCase(
        name="large / predictions on X_test",
        runner=lambda m: _run(m, _F3, lr=0.005, epochs=1000, output_idx=2),
        weight=2.0, atol=1e-6, rtol=1e-6,
        description="8 features, 1000 epochs — compare test predictions.",
    ),
]
