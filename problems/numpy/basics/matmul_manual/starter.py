"""手撕 matmul / 转置 / batched matmul。"""

from __future__ import annotations

import numpy as np


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # TODO: 矩阵乘 (M, K) @ (K, N) -> (M, N)
    # 禁用 np.dot / np.matmul / @ / np.einsum
    raise NotImplementedError


def transpose_2d(A: np.ndarray) -> np.ndarray:
    # TODO: 二维转置，禁用 .T / np.transpose / np.swapaxes
    raise NotImplementedError


def batched_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # TODO: batched matmul (B, M, K) @ (B, K, N) -> (B, M, N)。允许用 np.einsum。
    raise NotImplementedError
