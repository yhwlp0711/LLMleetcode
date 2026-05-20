"""参考实现：KMeans Lloyd 算法。"""

from __future__ import annotations

import numpy as np


def _pairwise_sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    # (N, K) 平方距离
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)


def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return _pairwise_sq_dist(X, centroids).argmin(axis=1).astype(np.int64)


def kmeans(
    X: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    centroids = init_centroids.copy()
    K = centroids.shape[0]

    for _ in range(max_iter):
        labels = _assign(X, centroids)
        new_centroids = centroids.copy()
        for k in range(K):
            mask = labels == k
            if mask.any():
                new_centroids[k] = X[mask].mean(axis=0)
            # 否则保留旧质心（空簇处理）

        shift = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if shift < tol:
            break

    labels = _assign(X, centroids)
    return centroids, labels
