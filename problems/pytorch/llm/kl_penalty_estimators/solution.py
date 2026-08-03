"""Reference: KL penalty estimators k1 / k2 / k3."""

from __future__ import annotations

import torch


def _logr(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    return logp_ref - logp


def kl_k1(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    return -_logr(logp, logp_ref)


def kl_k2(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    logr = _logr(logp, logp_ref)
    return 0.5 * logr.pow(2)


def kl_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    logr = _logr(logp, logp_ref)
    return torch.exp(logr) - 1.0 - logr
