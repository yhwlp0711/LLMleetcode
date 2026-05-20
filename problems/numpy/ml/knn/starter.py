"""KNN 分类。"""

from __future__ import annotations

import numpy as np


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    num_classes: int,
) -> np.ndarray:
    # TODO:
    # 1. 算 (M, N) 的成对平方距离矩阵
    # 2. argpartition / argsort 取每行前 k 个最近邻
    # 3. 对每个测试点的 k 个邻居标签做投票（np.bincount(minlength=num_classes)），
    #    argmax 取众数；并列时 argmax 默认返回最小索引，符合要求
    raise NotImplementedError
