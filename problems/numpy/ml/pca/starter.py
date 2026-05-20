"""PCA 主成分分析。"""

from __future__ import annotations

import numpy as np


def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO:
    # 1. 中心化: X_c = X - X.mean(axis=0)
    # 2. SVD:    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    # 3. 取前 k: components = Vt[:n_components]; explained_var = S[:n_components]**2 / (N - 1)
    # 4. 投影:   projected = X_c @ components.T
    # 5. 符号统一: 每个主成分的第一个非零元素应为正
    raise NotImplementedError
