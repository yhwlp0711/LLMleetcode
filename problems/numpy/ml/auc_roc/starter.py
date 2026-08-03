"""ROC 曲线与 AUC 计算。"""

from __future__ import annotations

import numpy as np


def auc_roc(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    # TODO: 计算 ROC 曲线并返回 (fpr, tpr, auc)。
    raise NotImplementedError
