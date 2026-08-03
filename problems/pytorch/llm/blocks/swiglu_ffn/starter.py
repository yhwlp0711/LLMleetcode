"""SwiGLU FFN（LLaMA 风格）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
