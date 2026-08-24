"""参考实现：LLaMA-style Transformer Block。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlleetcode.reference import rms_norm, sdpa


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, norm_eps: float = 1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm_eps = norm_eps

        self.attn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.ffn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, D = x.shape

        # --- Self-Attention sub-block (pre-norm) ---
        h = rms_norm(x, self.attn_norm_weight, self.norm_eps)
        q = self.W_q(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = sdpa(q, k, v, mask)  # (B, H, T, d_h)
        attn_out = self.W_o(attn_out.transpose(1, 2).reshape(B, T, D))
        x = x + attn_out

        # --- FFN sub-block (pre-norm) ---
        h = rms_norm(x, self.ffn_norm_weight, self.norm_eps)
        ffn_out = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + ffn_out
