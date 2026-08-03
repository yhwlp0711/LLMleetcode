"""激活函数 —— 按数学定义实现。

禁止调用 F.gelu / F.silu / nn.GELU / nn.SiLU。
"""

from __future__ import annotations

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    # TODO: 实现 SiLU（见 README）。
    raise NotImplementedError


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    # TODO: 实现精确版 GELU（用 erf，见 README）。
    raise NotImplementedError


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    # TODO: 实现 GELU 的 tanh 近似（见 README）。
    raise NotImplementedError
