"""Reference: DPO loss (forward)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    delta_chosen = policy_chosen_logps - ref_chosen_logps
    delta_rejected = policy_rejected_logps - ref_rejected_logps
    logits = beta * (delta_chosen - delta_rejected)
    loss = -F.logsigmoid(logits)
    return loss.mean()
