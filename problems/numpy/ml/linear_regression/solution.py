"""Reference solution for 001 Linear Regression."""
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
        y_hat = X_train @ w + b
        error = y_hat - y_train
        grad_w = (2.0 / N) * (X_train.T @ error)
        grad_b = (2.0 / N) * error.sum()
        w -= lr * grad_w
        b -= lr * grad_b

    y_pred = X_test @ w + b
    return w, float(b), y_pred
