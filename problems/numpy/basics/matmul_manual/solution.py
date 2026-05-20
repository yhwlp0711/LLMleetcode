"""Reference: Matmul, Transpose, Batched Matmul."""

from __future__ import annotations

import numpy as np


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    out = np.empty((M, N), dtype=np.result_type(A, B))
    # Vectorized per output row: out[i, :] = sum_k A[i, k] * B[k, :]
    for i in range(M):
        out[i, :] = (A[i, :, None] * B).sum(axis=0)
    return out


def transpose_2d(A: np.ndarray) -> np.ndarray:
    M, N = A.shape
    out = np.empty((N, M), dtype=A.dtype)
    # Fancy indexing — equivalent to .T
    rows = np.arange(M)[:, None]
    cols = np.arange(N)[None, :]
    out[cols, rows] = A[rows, cols]
    return out


def batched_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.einsum("bmk,bkn->bmn", A, B)
