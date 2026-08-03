"""Test cases for pytorch.llm.attention.mha."""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_mha")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _make_weights(D: int, seed: int):
    g = _g(seed)
    # Xavier-like init (1/sqrt(D)) to keep values well-conditioned.
    scale = 1.0 / (D**0.5)
    return [scale * torch.randn(D, D, generator=g) for _ in range(4)]


def _fx_basic():
    B, T, D, H = 1, 6, 16, 4
    x = torch.randn(B, T, D, generator=_g(0))
    W_q, W_k, W_v, W_o = _make_weights(D, seed=1)
    return x, W_q, W_k, W_v, W_o, H, None


def _fx_multihead_no_mask():
    B, T, D, H = 2, 8, 32, 8
    x = torch.randn(B, T, D, generator=_g(10))
    W_q, W_k, W_v, W_o = _make_weights(D, seed=11)
    return x, W_q, W_k, W_v, W_o, H, None


def _fx_causal():
    B, T, D, H = 2, 6, 24, 4
    x = torch.randn(B, T, D, generator=_g(20))
    W_q, W_k, W_v, W_o = _make_weights(D, seed=21)
    causal = (
        torch.tril(torch.ones(T, T, dtype=torch.bool))
        .view(1, 1, T, T)
        .expand(B, H, T, T)
    )
    return x, W_q, W_k, W_v, W_o, H, causal


def _fx_padding():
    B, T, D, H = 1, 7, 16, 4
    x = torch.randn(B, T, D, generator=_g(30))
    W_q, W_k, W_v, W_o = _make_weights(D, seed=31)
    keep = torch.tensor([True, True, True, True, True, False, False])
    pad_mask = keep.view(1, 1, 1, T).expand(B, H, T, T)
    return x, W_q, W_k, W_v, W_o, H, pad_mask


def _run(user_module, fixture):
    actual = user_module.mha(*fixture)
    expected = _REF.mha(*fixture)
    return actual, expected


TEST_CASES = [
    TestCase(
        name="single-batch / small head dim",
        runner=lambda m: _run(m, _fx_basic()),
        weight=1.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="multi-head, no mask",
        runner=lambda m: _run(m, _fx_multihead_no_mask()),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="causal mask",
        runner=lambda m: _run(m, _fx_causal()),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="padding mask",
        runner=lambda m: _run(m, _fx_padding()),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
]
