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
    # TODO: 对每个测试点找 k 个最近邻，多数投票得到预测类别。
    # 并列时取较小的类别索引。
    raise NotImplementedError
