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
        # TODO:
        # 1. z = X_train @ w + b
        # 2. p = sigmoid(z)  （用数值稳定的写法）
        # 3. grad_w = (1/N) * X_train.T @ (p - y_train)
        # 4. grad_b = (1/N) * (p - y_train).sum()
        # 5. w -= lr * grad_w ; b -= lr * grad_b
        raise NotImplementedError

    # TODO: 在 X_test 上预测概率
    proba_test = ...
    return w, float(b), proba_test
