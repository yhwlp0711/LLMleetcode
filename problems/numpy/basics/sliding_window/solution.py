"""Reference: Sliding Window Views."""

from __future__ import annotations

import numpy as np


def sliding_window_1d(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    full = np.lib.stride_tricks.sliding_window_view(x, window_shape=window)
    return np.ascontiguousarray(full[::stride])


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    w = sliding_window_1d(x, window=window, stride=1)
    return w.mean(axis=1)


def conv1d_valid(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    K = kernel.shape[0]
    w = sliding_window_1d(x, window=K, stride=1)  # shape (L-K+1, K)
    return w @ kernel
