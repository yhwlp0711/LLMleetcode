"""Test cases for Scaled Dot-Product Attention.

Demonstrates the "operator problem" template:
- judge constructs the inputs (q, k, v, mask) with fixed seed
- both user and reference receive the *same* tensors
- pure numeric comparison
"""
from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_sdpa")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _causal_mask(t_q: int, t_k: int, batch: int = 1, heads: int = 1) -> torch.Tensor:
    # True = keep (lower triangular incl. diagonal)
    m = torch.tril(torch.ones(t_q, t_k, dtype=torch.bool))
    return m.view(1, 1, t_q, t_k).expand(batch, heads, -1, -1)


def _fx_small():
    B, H, T, D = 1, 1, 4, 8
    q = torch.randn(B, H, T, D, generator=_g(0))
    k = torch.randn(B, H, T, D, generator=_g(1))
    v = torch.randn(B, H, T, D, generator=_g(2))
    return q, k, v


def _fx_multihead():
    B, H, T_q, T_k, D, D_v = 2, 4, 6, 8, 16, 12
    q = torch.randn(B, H, T_q, D, generator=_g(10))
    k = torch.randn(B, H, T_k, D, generator=_g(11))
    v = torch.randn(B, H, T_k, D_v, generator=_g(12))
    return q, k, v


def _fx_causal():
    B, H, T, D = 2, 2, 5, 8
    q = torch.randn(B, H, T, D, generator=_g(20))
    k = torch.randn(B, H, T, D, generator=_g(21))
    v = torch.randn(B, H, T, D, generator=_g(22))
    mask = _causal_mask(T, T, batch=B, heads=H)
    return q, k, v, mask


def _fx_padding():
    # Pad mask: last 3 positions of K are padding
    B, H, T_q, T_k, D = 1, 2, 4, 7, 8
    q = torch.randn(B, H, T_q, D, generator=_g(30))
    k = torch.randn(B, H, T_k, D, generator=_g(31))
    v = torch.randn(B, H, T_k, D, generator=_g(32))
    keep = torch.tensor([True, True, True, True, False, False, False])
    mask = keep.view(1, 1, 1, T_k).expand(B, H, T_q, T_k)
    return q, k, v, mask


TEST_CASES = [
    TestCase(
        name="no mask / single head",
        runner=lambda m: (m.sdpa(*_fx_small()), _REF.sdpa(*_fx_small())),
        weight=1.0, atol=1e-5, rtol=1e-5,
    ),
    TestCase(
        name="no mask / multi-head, T_q != T_k, D_v != D",
        runner=lambda m: (m.sdpa(*_fx_multihead()), _REF.sdpa(*_fx_multihead())),
        weight=2.0, atol=1e-5, rtol=1e-5,
    ),
    TestCase(
        name="causal mask",
        runner=lambda m: (m.sdpa(*_fx_causal()), _REF.sdpa(*_fx_causal())),
        weight=2.0, atol=1e-5, rtol=1e-5,
    ),
    TestCase(
        name="padding mask",
        runner=lambda m: (m.sdpa(*_fx_padding()), _REF.sdpa(*_fx_padding())),
        weight=2.0, atol=1e-5, rtol=1e-5,
    ),
]
