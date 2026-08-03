"""参考实现：LLaMA-style Transformer Block。"""

from __future__ import annotations

import torch
import torch.nn as nn

from mlleetcode.reference import rms_norm, sdpa, swiglu_ffn_forward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, norm_eps: float = 1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm_eps = norm_eps

        self.attn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.W_q = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_k = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_v = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_o = nn.Parameter(torch.zeros(d_model, d_model))

        self.ffn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.gate_proj = nn.Parameter(torch.zeros(d_model, d_ff))
        self.up_proj = nn.Parameter(torch.zeros(d_model, d_ff))
        self.down_proj = nn.Parameter(torch.zeros(d_ff, d_model))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, D = x.shape

        # --- Self-Attention sub-block (pre-norm) ---
        h = rms_norm(x, self.attn_norm_weight, self.norm_eps)
        q = (h @ self.W_q).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = (h @ self.W_k).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = (h @ self.W_v).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = sdpa(q, k, v, mask)  # (B, H, T, d_h)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D) @ self.W_o
        x = x + attn_out

        # --- FFN sub-block (pre-norm) ---
        h = rms_norm(x, self.ffn_norm_weight, self.norm_eps)
        ffn_out = swiglu_ffn_forward(h, self.gate_proj, self.up_proj, self.down_proj)
        return x + ffn_out
