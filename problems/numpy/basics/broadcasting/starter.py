"""广播与外积运算 —— 用 NumPy 实现下面四个函数。"""

from __future__ import annotations

import numpy as np


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # TODO: 用广播实现 R[i, j] = a[i] + b[j]
    raise NotImplementedError


def pairwise_difference(x: np.ndarray) -> np.ndarray:
    # TODO: 返回 D[i, j] = x[i] - x[j]
    raise NotImplementedError


def normalize_columns(X: np.ndarray) -> np.ndarray:
    # TODO: 按列做 z-score 归一化，使用总体方差（ddof=0）
    raise NotImplementedError


def apply_per_row_scale(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    # TODO: 第 i 行乘以标量 s[i]
    raise NotImplementedError
