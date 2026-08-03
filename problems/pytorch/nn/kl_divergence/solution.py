"""Reference: forward KL(P ‖ Q) from logits, numerically stable."""

from __future__ import annotations

import torch


def kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    log_p = p_logits - torch.logsumexp(p_logits, dim=-1, keepdim=True)
    log_q = q_logits - torch.logsumexp(q_logits, dim=-1, keepdim=True)
    p = log_p.exp()
    kl_per_sample = (p * (log_p - log_q)).sum(dim=-1)  # (N,)
    return kl_per_sample.mean()
