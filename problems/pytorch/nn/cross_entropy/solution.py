"""Reference: Cross-Entropy from logits (stable, with ignore_index)."""

from __future__ import annotations

import torch


def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    # Numerically stable log-softmax: z - logsumexp(z)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # (N, C)

    valid = target != ignore_index  # (N,)
    # Clamp ignored targets to a valid index so gather doesn't error; masked out below.
    safe_target = target.clone()
    safe_target[~valid] = 0
    picked = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)  # (N,)

    nll = -picked
    nll = nll[valid]
    return nll.mean()
