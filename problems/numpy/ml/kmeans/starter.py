"""KMeans 聚类 —— Lloyd 算法。"""

from __future__ import annotations

import numpy as np


def kmeans(
    X: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    centroids = init_centroids.copy()
    K = centroids.shape[0]

    for _ in range(max_iter):
        # TODO:
        # 1. 分配：对每个样本算到每个质心的平方距离 (N, K)，argmin 得 labels (N,)
        # 2. 更新：每个簇的新质心 = 该簇样本均值；空簇保留旧质心
        # 3. 收敛检查：max(|new_centroids - old_centroids|) < tol 就提前结束
        raise NotImplementedError

    # TODO: 最后再算一次 labels（基于最终质心）
    labels = ...
    return centroids, labels
