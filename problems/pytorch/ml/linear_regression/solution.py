"""Reference solution: Linear Regression with PyTorch Autograd."""

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
    _, D = X_train.shape
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros((), requires_grad=True)

    for _ in range(epochs):
        y_hat = X_train @ w + b
        loss = ((y_hat - y_train) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

    with torch.no_grad():
        y_pred = X_test @ w + b
    return w.detach(), float(b.item()), y_pred.detach()
