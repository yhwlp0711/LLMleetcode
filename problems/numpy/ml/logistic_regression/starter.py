"""逻辑回归 —— 实现训练与概率预测。"""

from __future__ import annotations

import numpy as np


def fit_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    N, D = X_train.shape
    w = np.zeros(D, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        # TODO: 一步批量梯度下降——sigmoid 前向（数值稳定）、算梯度、更新 w 和 b
        raise NotImplementedError

    # TODO: 在 X_test 上预测概率
    proba_test = ...
    return w, float(b), proba_test
