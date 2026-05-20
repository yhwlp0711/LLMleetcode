"""Test cases for pytorch.llm.transformer_block."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult
from mlleetcode.utils.sandbox import load_module_from_path
from mlleetcode.utils.stats import check_shape
from mlleetcode.utils.weights import assert_param_names, sync_weights

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_block")


def _g(seed):
    return torch.Generator().manual_seed(seed)


EXPECTED_PARAMS = [
    "attn_norm_weight",
    "W_q",
    "W_k",
    "W_v",
    "W_o",
    "ffn_norm_weight",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _check_init(user_module) -> CompareResult:
    d_model, num_heads, d_ff = 32, 4, 64
    mod = user_module.TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    ok, msg = assert_param_names(mod, EXPECTED_PARAMS)
    if not ok:
        return CompareResult(passed=False, reason=f"params wrong: {msg}")

    expected_shapes = {
        "attn_norm_weight": (d_model,),
        "W_q": (d_model, d_model),
        "W_k": (d_model, d_model),
        "W_v": (d_model, d_model),
        "W_o": (d_model, d_model),
        "ffn_norm_weight": (d_model,),
        "gate_proj": (d_model, d_ff),
        "up_proj": (d_model, d_ff),
        "down_proj": (d_ff, d_model),
    }
    for name, shape in expected_shapes.items():
        t = getattr(mod, name)
        s = check_shape(t, shape, name=name)
        if not s.passed:
            return s
    return CompareResult(passed=True)


def _run_forward(user_module, d_model, num_heads, d_ff, shape, seed, with_mask):
    user_mod = user_module.TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    ref_mod = _REF.TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    # 用随机权重覆盖参考（默认全 0 太特殊）
    g = _g(seed + 1000)
    with torch.no_grad():
        scale = 1.0 / (d_model**0.5)
        ref_mod.W_q.copy_(scale * torch.randn(d_model, d_model, generator=g))
        ref_mod.W_k.copy_(scale * torch.randn(d_model, d_model, generator=g))
        ref_mod.W_v.copy_(scale * torch.randn(d_model, d_model, generator=g))
        ref_mod.W_o.copy_(scale * torch.randn(d_model, d_model, generator=g))
        ref_mod.gate_proj.copy_(scale * torch.randn(d_model, d_ff, generator=g))
        ref_mod.up_proj.copy_(scale * torch.randn(d_model, d_ff, generator=g))
        ref_mod.down_proj.copy_(scale * torch.randn(d_ff, d_model, generator=g))
    sync_weights(user_mod, ref_mod)

    x = torch.randn(*shape, generator=_g(seed))
    mask = None
    if with_mask:
        T = shape[1]
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)
    return user_mod(x.clone(), mask), ref_mod(x.clone(), mask)


TEST_CASES = [
    TestCase(name="init / params + shapes", runner=_check_init, weight=2.0),
    TestCase(
        name="forward / small, no mask",
        runner=lambda m: _run_forward(m, 32, 4, 64, (2, 6, 32), 10, False),
        weight=3.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="forward / causal mask",
        runner=lambda m: _run_forward(m, 32, 4, 64, (1, 8, 32), 20, True),
        weight=3.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="forward / larger dims",
        runner=lambda m: _run_forward(m, 48, 6, 96, (2, 8, 48), 30, False),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
]
