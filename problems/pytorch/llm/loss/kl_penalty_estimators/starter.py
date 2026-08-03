"""KL 惩罚估计器 —— John Schulman 的 k1 / k2 / k3。

logr = logp_ref - logp
"""

from __future__ import annotations

import torch


def kl_k1(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # TODO: 实现 k1 估计器（见 README）。
    raise NotImplementedError


def kl_k2(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # TODO: 实现 k2 估计器（见 README）。
    raise NotImplementedError


def kl_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # TODO: 实现 k3 估计器（见 README）。
    raise NotImplementedError
