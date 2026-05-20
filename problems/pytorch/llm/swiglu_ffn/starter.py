"""SwiGLU FFN（LLaMA 风格）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # TODO: 创建 gate_proj、up_proj、down_proj 三个 nn.Linear(..., bias=False)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: down_proj(silu(gate_proj(x)) * up_proj(x))
        raise NotImplementedError
