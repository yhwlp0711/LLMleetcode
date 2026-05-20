"""Test cases for numpy.ml.auc_roc."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult, compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_auc")


def _rng(seed):
    return np.random.default_rng(seed)


def _make(seed, N, noise: float = 0.5):
    rng = _rng(seed)
    y_true = (rng.standard_normal(N) > 0).astype(np.int64)
    y_score = y_true + noise * rng.standard_normal(N)
    return y_true, y_score


def _run(user_module, y_true, y_score, idx):
    user_out = user_module.auc_roc(y_true.copy(), y_score.copy())
    ref_out = _REF.auc_roc(y_true.copy(), y_score.copy())
    return user_out[idx], ref_out[idx]


def _check_perfect_predictor(user_module) -> CompareResult:
    """完美预测器 (score == label) AUC 必须是 1.0"""
    y_true = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    y_score = y_true.astype(np.float64)
    _, _, auc = user_module.auc_roc(y_true, y_score)
    if abs(auc - 1.0) > 1e-10:
        return CompareResult(
            passed=False, reason=f"perfect predictor should give AUC=1.0, got {auc}"
        )
    return CompareResult(passed=True)


def _check_random_predictor(user_module) -> CompareResult:
    """逆序预测器：score = -y。AUC 应该是 0.0（最差预测）。"""
    y_true = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    y_score = -y_true.astype(np.float64)  # 1->-1, 0->0；正样本反而排在最后
    _, _, auc = user_module.auc_roc(y_true, y_score)
    if abs(auc - 0.0) > 1e-10:
        return CompareResult(
            passed=False, reason=f"inverse predictor should give AUC=0.0, got {auc}"
        )
    return CompareResult(passed=True)


_yt1, _ys1 = _make(seed=0, N=100, noise=0.5)
_yt2, _ys2 = _make(seed=1, N=500, noise=1.0)


TEST_CASES = [
    TestCase(
        name="medium / fpr",
        runner=lambda m: _run(m, _yt1, _ys1, 0),
        weight=1.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="medium / tpr",
        runner=lambda m: _run(m, _yt1, _ys1, 1),
        weight=1.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="medium / auc",
        runner=lambda m: _run(m, _yt1, _ys1, 2),
        weight=2.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="larger / auc",
        runner=lambda m: _run(m, _yt2, _ys2, 2),
        weight=2.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="property / perfect predictor AUC=1.0",
        runner=_check_perfect_predictor,
        weight=1.0,
    ),
    TestCase(
        name="property / inverse predictor AUC=0.0",
        runner=_check_random_predictor,
        weight=1.0,
    ),
]
