"""Reference: Rotary Position Embeddings (RoPE)."""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    # theta_i = 1 / base^(2i / head_dim), i in [0, half)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32) * 2 / head_dim)
    )
    pos = torch.arange(seq_len, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]  # (seq_len, half)
    cos = angles.cos()
    sin = angles.sin()
    # Duplicate each pair so cos[m, 2i] == cos[m, 2i+1].
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, head_dim); cos/sin: (T, head_dim) -> broadcast to (1, 1, T, head_dim)
    B, H, T, D = x.shape
    cos_b = cos.view(1, 1, T, D)
    sin_b = sin.view(1, 1, T, D)

    # Build x_rotated: pair (a, b) -> (-b, a) at every adjacent (2i, 2i+1)
    x_pairs = x.reshape(B, H, T, D // 2, 2)
    a = x_pairs[..., 0]
    b = x_pairs[..., 1]
    x_rotated = torch.stack([-b, a], dim=-1).reshape(B, H, T, D)

    return x * cos_b + x_rotated * sin_b
