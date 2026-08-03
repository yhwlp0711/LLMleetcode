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
        # TODO: 一轮 Lloyd 迭代——分配样本到最近质心、更新质心；
        # 空簇保留旧质心；质心几乎不变（< tol）时提前结束。
        raise NotImplementedError

    # TODO: 用最终质心再算一次每个样本的 labels
    labels = ...
    return centroids, labels
