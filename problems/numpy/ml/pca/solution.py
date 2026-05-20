"""参考实现：PCA。"""

from __future__ import annotations

import numpy as np


def _fix_sign(components: np.ndarray, projected: np.ndarray) -> None:
    """In-place: ensure first non-zero entry of each component is positive."""
    eps = 1e-12
    for i in range(components.shape[0]):
        row = components[i]
        # find first index with |row| > eps
        nonzero = np.abs(row) > eps
        if not nonzero.any():
            continue
        first = np.argmax(nonzero)  # 第一个 True 的索引
        if row[first] < 0:
            components[i] = -components[i]
            projected[:, i] = -projected[:, i]


def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = X.shape[0]
    X_c = X - X.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    components = Vt[:n_components].copy()  # (k, D)
    explained_var = (S[:n_components] ** 2) / (N - 1)  # (k,)
    projected = X_c @ components.T  # (N, k)

    _fix_sign(components, projected)
    return components, explained_var, projected
