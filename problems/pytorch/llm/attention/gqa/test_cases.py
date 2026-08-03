"""Test cases for pytorch.llm.attention.gqa."""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_gqa")


def _g(seed):
    return torch.Generator().manual_seed(seed)


def _make(B, T, D, num_q_heads, num_kv_heads, seed):
    g = _g(seed)
    head_dim = D // num_q_heads
    scale = 1.0 / (D**0.5)
    x = torch.randn(B, T, D, generator=g)
    W_q = scale * torch.randn(D, num_q_heads * head_dim, generator=g)
    W_k = scale * torch.randn(D, num_kv_heads * head_dim, generator=g)
    W_v = scale * torch.randn(D, num_kv_heads * head_dim, generator=g)
    W_o = scale * torch.randn(num_q_heads * head_dim, D, generator=g)
    return x, W_q, W_k, W_v, W_o


def _run(user_module, args, num_q_heads, num_kv_heads, mask=None):
    return (
        user_module.gqa(
            *args, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, mask=mask
        ),
        _REF.gqa(*args, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, mask=mask),
    )


# Fixture: D=32, head_dim=8, 4 q-heads
_A1 = _make(B=2, T=6, D=32, num_q_heads=4, num_kv_heads=2, seed=0)
# Fixture: 8 q-heads, 4 kv-heads (典型 GQA)
_A2 = _make(B=1, T=8, D=64, num_q_heads=8, num_kv_heads=4, seed=1)
# Fixture: MQA 退化 (num_kv_heads=1)
_A3 = _make(B=1, T=6, D=32, num_q_heads=4, num_kv_heads=1, seed=2)
# Fixture: MHA 退化 (num_kv_heads == num_q_heads)
_A4 = _make(B=1, T=6, D=32, num_q_heads=4, num_kv_heads=4, seed=3)


def _causal(B, H, T):
    return (
        torch.tril(torch.ones(T, T, dtype=torch.bool))
        .view(1, 1, T, T)
        .expand(B, H, T, T)
    )


TEST_CASES = [
    TestCase(
        name="GQA / 4-q 2-kv heads, no mask",
        runner=lambda m: _run(m, _A1, 4, 2),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="GQA / 8-q 4-kv heads, no mask",
        runner=lambda m: _run(m, _A2, 8, 4),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="MQA (num_kv_heads=1) degeneration",
        runner=lambda m: _run(m, _A3, 4, 1),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="MHA (num_kv_heads=num_q_heads) degeneration",
        runner=lambda m: _run(m, _A4, 4, 4),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="GQA with causal mask",
        runner=lambda m: _run(m, _A1, 4, 2, mask=_causal(2, 4, 6)),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
]
