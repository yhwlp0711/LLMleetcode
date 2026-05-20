"""参考实现：Sinusoidal Position Encoding。"""

from __future__ import annotations

import torch


def build_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    )
    pos = torch.arange(seq_len, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]  # (seq_len, d_model/2)

    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = angles.sin()
    pe[:, 1::2] = angles.cos()
    return pe
