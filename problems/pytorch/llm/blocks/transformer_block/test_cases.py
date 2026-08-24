"""Test cases for pytorch.llm.blocks.transformer_block."""

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
    "W_q.weight",
    "W_k.weight",
    "W_v.weight",
    "W_o.weight",
    "ffn_norm_weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
]

# nn.Linear 权重形状为 (out_features, in_features)
LINEAR_SHAPES = {
    "W_q": (32, 32),
    "W_k": (32, 32),
    "W_v": (32, 32),
    "W_o": (32, 32),
    "gate_proj": (64, 32),
    "up_proj": (64, 32),
    "down_proj": (32, 64),
}


def _check_init(user_module) -> CompareResult:
    d_model, num_heads, d_ff = 32, 4, 64
    mod = user_module.TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    ok, msg = assert_param_names(mod, EXPECTED_PARAMS)
    if not ok:
        return CompareResult(passed=False, reason=f"params wrong: {msg}")

    for name, shape in [
        ("attn_norm_weight", (d_model,)),
        ("ffn_norm_weight", (d_model,)),
    ]:
        s = check_shape(getattr(mod, name), shape, name=name)
        if not s.passed:
            return s

    for name, shape in LINEAR_SHAPES.items():
        layer = getattr(mod, name)
        if not isinstance(layer, nn.Linear):
            return CompareResult(passed=False, reason=f"{name} is not nn.Linear")
        if layer.bias is not None:
            return CompareResult(
                passed=False, reason=f"{name} has bias (should be bias=False)"
            )
        s = check_shape(layer.weight, shape, name=f"{name}.weight")
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
    # 用随机权重覆盖参考（默认权重太小/接近 0，区分度不足）
    # nn.Linear.weight 形状为 (out_features, in_features)
    g = _g(seed + 1000)
    with torch.no_grad():
        scale = 1.0 / (d_model**0.5)
        ref_mod.W_q.weight.copy_(scale * torch.randn(d_model, d_model, generator=g))
        ref_mod.W_k.weight.copy_(scale * torch.randn(d_model, d_model, generator=g))
        ref_mod.W_v.weight.copy_(scale * torch.randn(d_model, d_model, generator=g))
        ref_mod.W_o.weight.copy_(scale * torch.randn(d_model, d_model, generator=g))
        ref_mod.gate_proj.weight.copy_(scale * torch.randn(d_ff, d_model, generator=g))
        ref_mod.up_proj.weight.copy_(scale * torch.randn(d_ff, d_model, generator=g))
        ref_mod.down_proj.weight.copy_(scale * torch.randn(d_model, d_ff, generator=g))
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
