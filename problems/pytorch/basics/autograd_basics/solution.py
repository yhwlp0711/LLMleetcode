"""Reference: Autograd Basics."""

from __future__ import annotations

from typing import Callable

import torch


def grad_of_scalar(
    x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    z = x.detach().clone().requires_grad_(True)
    y = fn(z)
    (grad,) = torch.autograd.grad(y, z, create_graph=False)
    return grad


def numerical_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    x0 = x.detach().clone()
    y0 = fn(x0.clone())
    n = x0.numel()
    m = y0.numel()
    J = torch.zeros(m, n, dtype=x0.dtype)
    flat = x0.reshape(-1)
    for j in range(n):
        e = torch.zeros_like(flat)
        e[j] = eps
        x_plus = (flat + e).reshape(x0.shape)
        x_minus = (flat - e).reshape(x0.shape)
        y_plus = fn(x_plus.clone()).reshape(-1)
        y_minus = fn(x_minus.clone()).reshape(-1)
        J[:, j] = (y_plus - y_minus) / (2 * eps)
    return J


def sgd_minimize(
    fn: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, lr: float, steps: int
) -> torch.Tensor:
    x = x0.detach().clone().requires_grad_(True)
    for _ in range(steps):
        y = fn(x)
        y.backward()
        with torch.no_grad():
            x -= lr * x.grad
            x.grad.zero_()
    return x.detach()
