"""张量操作热身 —— 用 PyTorch 实现下面 5 个函数。"""

from __future__ import annotations

import torch


def flatten_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: 把两个张量都展平后拼成一个一维张量
    raise NotImplementedError


def row_softmax(x: torch.Tensor) -> torch.Tensor:
    # TODO: 沿最后一维做数值稳定的 softmax
    raise NotImplementedError


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # TODO: 返回 (N, M) 的成对平方距离矩阵，禁用 Python 循环
    raise NotImplementedError


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # TODO: 沿时间轴对 mask=True 的位置求均值。输出 shape (B, D)。
    raise NotImplementedError


def top_k_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    # TODO: top-k 值的索引，按分数降序；并列时索引较小者优先
    raise NotImplementedError
