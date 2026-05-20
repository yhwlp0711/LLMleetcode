"""Test cases for pytorch.ml.linear_regression."""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_pt_lr")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _make_dataset(seed: int, n_train: int, n_test: int, d: int):
    g = _g(seed)
    X_train = torch.randn(n_train, d, generator=g)
    true_w = torch.randn(d, generator=g)
    true_b = torch.randn((), generator=g)
    noise = 0.1 * torch.randn(n_train, generator=g)
    y_train = X_train @ true_w + true_b + noise
    X_test = torch.randn(n_test, d, generator=g)
    return X_train, y_train, X_test


_F1 = _make_dataset(seed=0, n_train=50, n_test=10, d=3)
_F2 = _make_dataset(seed=1, n_train=200, n_test=40, d=5)
_F3 = _make_dataset(seed=2, n_train=500, n_test=100, d=8)


def _run(user_module, fixture, lr, epochs, idx):
    X_train, y_train, X_test = fixture
    actual = user_module.fit_predict(
        X_train.clone(),
        y_train.clone(),
        X_test.clone(),
        lr=lr,
        epochs=epochs,
    )
    expected = _REF.fit_predict(
        X_train.clone(),
        y_train.clone(),
        X_test.clone(),
        lr=lr,
        epochs=epochs,
    )
    return actual[idx], expected[idx]


TEST_CASES = [
    TestCase(
        name="small / weights w",
        runner=lambda m: _run(m, _F1, lr=0.05, epochs=200, idx=0),
        weight=1.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="small / bias b",
        runner=lambda m: _run(m, _F1, lr=0.05, epochs=200, idx=1),
        weight=1.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="medium / predictions",
        runner=lambda m: _run(m, _F2, lr=0.01, epochs=500, idx=2),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="large / predictions",
        runner=lambda m: _run(m, _F3, lr=0.005, epochs=1000, idx=2),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
]
