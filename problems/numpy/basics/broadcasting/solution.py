"""Reference solution: Broadcasting & Outer Operations."""

from __future__ import annotations

import numpy as np


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, None] + b[None, :]


def pairwise_difference(x: np.ndarray) -> np.ndarray:
    return x[:, None] - x[None, :]


def normalize_columns(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=0, keepdims=True)
    return (X - mean) / std


def apply_per_row_scale(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    return s[:, None] * X
