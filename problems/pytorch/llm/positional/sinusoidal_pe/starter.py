"""Sinusoidal Position Encoding。"""

from __future__ import annotations

import torch


def build_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: 返回 (seq_len, d_model) 的正弦位置编码表：偶数维放 sin、奇数维放 cos，频率随维度递减。
    raise NotImplementedError
