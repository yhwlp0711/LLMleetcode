"""线性回归（PyTorch Autograd 版）。"""

from __future__ import annotations

import torch


def fit_predict(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    lr: float,
    epochs: int,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    N, D = X_train.shape
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros((), requires_grad=True)

    for _ in range(epochs):
        # TODO: 前向 → 计算 loss → backward → 在 no_grad 下手动 SGD 更新 → 清梯度
        raise NotImplementedError

    with torch.no_grad():
        y_pred = X_test @ w + b
    return w.detach(), float(b.item()), y_pred.detach()
