"""Reference: Logistic Regression from Scratch."""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable: avoid overflow of exp(-z) for very negative z.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


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
        z = X_train @ w + b
        p = _sigmoid(z)
        err = p - y_train
        w -= lr * (X_train.T @ err) / N
        b -= lr * err.sum() / N

    proba_test = _sigmoid(X_test @ w + b)
    return w, float(b), proba_test
