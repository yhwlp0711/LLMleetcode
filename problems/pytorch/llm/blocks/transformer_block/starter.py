"""LLaMA-style Transformer Block。"""

from __future__ import annotations

import torch
import torch.nn as nn

from mlleetcode.reference import rms_norm, sdpa, swiglu_ffn_forward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, norm_eps: float = 1e-6):
        super().__init__()
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError
