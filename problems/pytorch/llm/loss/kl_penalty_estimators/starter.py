"""KL 惩罚估计器 —— John Schulman 的 k1 / k2 / k3。

logr = logp_ref - logp
"""

from __future__ import annotations

import torch


def kl_k1(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # TODO: k1 = -logr = logp - logp_ref
    raise NotImplementedError


def kl_k2(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # TODO: k2 = 0.5 * logr**2
    raise NotImplementedError


def kl_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # TODO: k3 = (exp(logr) - 1) - logr
    raise NotImplementedError
