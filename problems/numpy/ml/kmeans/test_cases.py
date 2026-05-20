"""Test cases for numpy.ml.kmeans."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_kmeans")


def _rng(seed: int):
    return np.random.default_rng(seed)


def _make_blobs(
    seed: int, n_per_cluster: int = 50, d: int = 4, k: int = 3, spread: float = 0.5
):
    rng = _rng(seed)
    centers = rng.standard_normal((k, d)) * 3.0
    X = np.vstack(
        [
            centers[c] + spread * rng.standard_normal((n_per_cluster, d))
            for c in range(k)
        ]
    )
    rng.shuffle(X)
    return X, centers


def _run(user_module, X, init, max_iter, idx):
    user_out = user_module.kmeans(X.copy(), init.copy(), max_iter=max_iter)
    ref_out = _REF.kmeans(X.copy(), init.copy(), max_iter=max_iter)
    return user_out[idx], ref_out[idx]


# Fixture 1: 干净的三簇
_X1, _C1 = _make_blobs(seed=0, n_per_cluster=40, d=4, k=3, spread=0.3)
_INIT1 = _C1 + 0.5  # 稍微偏离真实中心

# Fixture 2: 较大维度
_X2, _C2 = _make_blobs(seed=1, n_per_cluster=30, d=8, k=5, spread=0.6)
_INIT2 = _C2 + 1.0

# Fixture 3: 制造空簇场景 —— 一个 init 质心远离所有数据
_X3, _C3 = _make_blobs(seed=2, n_per_cluster=30, d=3, k=2, spread=0.2)
_INIT3 = np.vstack([_C3, np.array([[100.0, 100.0, 100.0]])])  # 第 3 个会一直是空簇


TEST_CASES = [
    TestCase(
        name="3-cluster / centroids",
        runner=lambda m: _run(m, _X1, _INIT1, 30, 0),
        weight=1.0,
        atol=1e-8,
        rtol=1e-8,
    ),
    TestCase(
        name="3-cluster / labels",
        runner=lambda m: _run(m, _X1, _INIT1, 30, 1),
        weight=1.0,
    ),
    TestCase(
        name="5-cluster / centroids",
        runner=lambda m: _run(m, _X2, _INIT2, 30, 0),
        weight=2.0,
        atol=1e-8,
        rtol=1e-8,
    ),
    TestCase(
        name="5-cluster / labels",
        runner=lambda m: _run(m, _X2, _INIT2, 30, 1),
        weight=2.0,
    ),
    TestCase(
        name="empty cluster / centroids (third stays at init)",
        runner=lambda m: _run(m, _X3, _INIT3, 30, 0),
        weight=2.0,
        atol=1e-8,
        rtol=1e-8,
    ),
]
