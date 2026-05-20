"""Autograd 基础。"""

from __future__ import annotations

from typing import Callable

import torch


def grad_of_scalar(
    x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    # TODO: 返回 df/dx，其中 f = fn(x)；不要修改 x。
    raise NotImplementedError


def numerical_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    # TODO: 用中心差分估计 Jacobian (m, n)。禁用 autograd！
    raise NotImplementedError


def sgd_minimize(
    fn: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, lr: float, steps: int
) -> torch.Tensor:
    # TODO: 从 x0 出发跑 steps 步 SGD。
    raise NotImplementedError
