"""参考实现：ROC 曲线与 AUC。"""

from __future__ import annotations

import numpy as np


def auc_roc(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    # 1. 降序排
    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order].astype(np.float64)

    # 2. 累加 TP / FP
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)

    P = y_sorted.sum()
    N = len(y_sorted) - P

    # 3. TPR / FPR + 头部 (0, 0)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])

    # 4. 梯形积分（手写：sum 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])）
    auc = float((0.5 * (tpr[1:] + tpr[:-1]) * np.diff(fpr)).sum())
    return fpr, tpr, auc
