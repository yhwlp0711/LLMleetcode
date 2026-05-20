"""Test cases for numpy.ml.logistic_regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_logreg")


def _make_dataset(seed: int, n_train: int, n_test: int, d: int):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, d))
    true_w = rng.standard_normal(d)
    true_b = rng.standard_normal() * 0.5
    logits = X_train @ true_w + true_b
    y_train = (logits + 0.3 * rng.standard_normal(n_train) > 0).astype(np.float64)
    X_test = rng.standard_normal((n_test, d))
    return X_train, y_train, X_test


_F1 = _make_dataset(seed=0, n_train=100, n_test=20, d=3)
_F2 = _make_dataset(seed=1, n_train=300, n_test=50, d=6)
_F3 = _make_dataset(seed=2, n_train=600, n_test=100, d=10)


def _run(user_module, fixture, lr, epochs, idx):
    X_train, y_train, X_test = fixture
    actual = user_module.fit_predict_proba(
        X_train.copy(),
        y_train.copy(),
        X_test.copy(),
        lr=lr,
        epochs=epochs,
    )
    expected = _REF.fit_predict_proba(
        X_train.copy(),
        y_train.copy(),
        X_test.copy(),
        lr=lr,
        epochs=epochs,
    )
    return actual[idx], expected[idx]


TEST_CASES = [
    TestCase(
        name="small / weights w",
        runner=lambda m: _run(m, _F1, 0.1, 300, 0),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="small / bias b",
        runner=lambda m: _run(m, _F1, 0.1, 300, 1),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="medium / proba on X_test",
        runner=lambda m: _run(m, _F2, 0.05, 500, 2),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="large / proba on X_test",
        runner=lambda m: _run(m, _F3, 0.03, 800, 2),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
]
