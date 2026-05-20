"""ROC 曲线与 AUC 计算。"""

from __future__ import annotations

import numpy as np


def auc_roc(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    # TODO:
    # 1. 按 y_score 降序排，y_true 跟着重排
    # 2. cumsum(y_true_sorted) 是 TP，cumsum(1 - y_true_sorted) 是 FP
    # 3. TPR = TP / P，FPR = FP / N
    # 4. 在头部加 (0, 0) 点
    # 5. AUC = trapezoidal area
    raise NotImplementedError
