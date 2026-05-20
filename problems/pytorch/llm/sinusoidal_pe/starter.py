"""Sinusoidal Position Encoding。"""

from __future__ import annotations

import torch


def build_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: 返回 (seq_len, d_model) 的 sin/cos 表
    # 提示：
    # 1. inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2) / d_model))   # (d_model/2,)
    # 2. pos = torch.arange(seq_len)
    # 3. angles = pos[:, None] * inv_freq[None, :]    # (seq_len, d_model/2)
    # 4. pe[:, 0::2] = sin(angles), pe[:, 1::2] = cos(angles)
    raise NotImplementedError
