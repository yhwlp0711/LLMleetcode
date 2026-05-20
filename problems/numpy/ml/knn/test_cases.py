"""Test cases for numpy.ml.knn."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_knn")


def _rng(seed: int):
    return np.random.default_rng(seed)


def _make_classification(seed: int, n_per_class: int, d: int, num_classes: int):
    rng = _rng(seed)
    centers = rng.standard_normal((num_classes, d)) * 3.0
    X = np.vstack(
        [
            centers[c] + 0.5 * rng.standard_normal((n_per_class, d))
            for c in range(num_classes)
        ]
    )
    y = np.repeat(np.arange(num_classes, dtype=np.int64), n_per_class)
    return X, y


def _split(X, y, n_test, seed):
    rng = _rng(seed)
    perm = rng.permutation(len(X))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


_X1, _y1 = _make_classification(seed=0, n_per_class=30, d=4, num_classes=3)
_Xtr1, _ytr1, _Xte1, _yte1 = _split(_X1, _y1, n_test=20, seed=100)

_X2, _y2 = _make_classification(seed=1, n_per_class=50, d=8, num_classes=5)
_Xtr2, _ytr2, _Xte2, _yte2 = _split(_X2, _y2, n_test=40, seed=101)


def _run(user_module, X_train, y_train, X_test, k, num_classes):
    return (
        user_module.knn_predict(
            X_train.copy(), y_train.copy(), X_test.copy(), k=k, num_classes=num_classes
        ),
        _REF.knn_predict(
            X_train.copy(), y_train.copy(), X_test.copy(), k=k, num_classes=num_classes
        ),
    )


TEST_CASES = [
    TestCase(
        name="k=1 / 3 classes",
        runner=lambda m: _run(m, _Xtr1, _ytr1, _Xte1, 1, 3),
        weight=1.0,
    ),
    TestCase(
        name="k=5 / 3 classes",
        runner=lambda m: _run(m, _Xtr1, _ytr1, _Xte1, 5, 3),
        weight=2.0,
    ),
    TestCase(
        name="k=7 / 5 classes",
        runner=lambda m: _run(m, _Xtr2, _ytr2, _Xte2, 7, 5),
        weight=2.0,
    ),
    TestCase(
        name="k=15 / 5 classes",
        runner=lambda m: _run(m, _Xtr2, _ytr2, _Xte2, 15, 5),
        weight=2.0,
    ),
]
