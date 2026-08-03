"""PCA 主成分分析。"""

from __future__ import annotations

import numpy as np


def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: 中心化数据 → SVD → 取前 n_components 个主成分，返回
    # (components, explained_variance, projected)。
    # 注意符号统一：每个主成分第一个非零元素为正（保证结果确定）。
    raise NotImplementedError
