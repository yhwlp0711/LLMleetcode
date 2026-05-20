"""参考实现：KNN 分类。"""

from __future__ import annotations

import numpy as np


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    num_classes: int,
) -> np.ndarray:
    # 平方距离矩阵 (M, N)，用 (a-b)^2 = a^2 - 2ab + b^2
    x2 = (X_test**2).sum(axis=1, keepdims=True)  # (M, 1)
    t2 = (X_train**2).sum(axis=1)  # (N,)
    xt = X_test @ X_train.T  # (M, N)
    dist = x2 - 2.0 * xt + t2  # (M, N)

    # 取前 k 个最近邻索引 —— 用 argpartition 加 argsort 保证稳定顺序
    M = dist.shape[0]
    idx_k = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]  # (M, k) 未排序
    # 投票
    preds = np.empty(M, dtype=np.int64)
    for i in range(M):
        votes = np.bincount(y_train[idx_k[i]], minlength=num_classes)
        preds[i] = votes.argmax()  # 并列时 argmax 取最小索引，符合要求
    return preds
