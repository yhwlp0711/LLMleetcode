"""手撕线性回归 —— 用批量梯度下降实现，详见 README.md。"""

from __future__ import annotations

import numpy as np


def fit_predict(
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
        # TODO: 实现前向、梯度、参数更新
        # 1. y_hat = X_train @ w + b
        # 2. error = y_hat - y_train
        # 3. grad_w = (2 / N) * X_train.T @ error
        # 4. grad_b = (2 / N) * error.sum()
        # 5. w -= lr * grad_w ; b -= lr * grad_b
        raise NotImplementedError("请实现训练循环")

    # TODO: 用训练好的 (w, b) 在 X_test 上预测
    y_pred = ...

    return w, float(b), y_pred
