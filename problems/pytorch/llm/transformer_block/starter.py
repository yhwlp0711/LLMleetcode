"""LLaMA-style Transformer Block。"""

from __future__ import annotations

import torch
import torch.nn as nn

from mlleetcode.reference import rms_norm, sdpa, swiglu_ffn_forward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, norm_eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm_eps = norm_eps

        # TODO: 创建以下参数（命名必须一致）：
        # self.attn_norm_weight = nn.Parameter(torch.ones(d_model))
        # self.W_q, self.W_k, self.W_v: nn.Parameter(torch.zeros(d_model, d_model))
        # self.W_o:                       nn.Parameter(torch.zeros(d_model, d_model))
        # self.ffn_norm_weight = nn.Parameter(torch.ones(d_model))
        # self.gate_proj, self.up_proj:  nn.Parameter(torch.zeros(d_model, d_ff))
        # self.down_proj:                 nn.Parameter(torch.zeros(d_ff, d_model))
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # TODO: pre-norm self-attention + residual; pre-norm SwiGLU FFN + residual
        # 1. h = rms_norm(x, self.attn_norm_weight, self.norm_eps)
        # 2. Q/K/V 投影 + 切多头 + sdpa + 合头 + W_o
        # 3. x = x + attn_out
        # 4. h = rms_norm(x, self.ffn_norm_weight, self.norm_eps)
        # 5. ffn_out = swiglu_ffn_forward(h, gate_proj, up_proj, down_proj)
        # 6. return x + ffn_out
        raise NotImplementedError
