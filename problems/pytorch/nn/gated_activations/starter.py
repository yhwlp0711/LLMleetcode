"""门控激活函数 —— 按数学定义实现。

禁止调用 F.silu / F.gelu / F.glu / nn.SiLU / nn.GELU。
"""

from __future__ import annotations

import torch


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # TODO: SwiGLU（见 README）。注意激活作用在 gate 上，再与 x 逐元素相乘。
    raise NotImplementedError


def geglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # TODO: GeGLU（见 README）。
    raise NotImplementedError
