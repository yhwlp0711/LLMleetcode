"""门控激活函数 —— 按数学定义实现。

禁止调用 F.silu / F.gelu / F.glu / nn.SiLU / nn.GELU。
"""

from __future__ import annotations

import torch


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # TODO: SiLU(gate) * x
    #   SiLU(z) = z * sigmoid(z)
    #   注意顺序：SiLU 作用在 gate 上，再与 x 逐元素相乘
    raise NotImplementedError


def geglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # TODO: GELU_exact(gate) * x
    #   GELU_exact(z) = 0.5 * z * (1 + erf(z / sqrt(2)))
    raise NotImplementedError
