"""Test cases for pytorch.llm.blocks.swiglu_ffn."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult
from mlleetcode.utils.sandbox import load_module_from_path
from mlleetcode.utils.stats import check_shape
from mlleetcode.utils.weights import assert_param_names, sync_weights

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_swiglu_ffn")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


# ---- Init checks (structure only; values are random per init) -------------


def _check_init(user_module) -> CompareResult:
    d_model, d_ff = 32, 64
    mod = user_module.SwiGLUFFN(d_model=d_model, d_ff=d_ff)

    # Expected param names: weights of three Linear(bias=False) layers
    expected = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
    ok, msg = assert_param_names(mod, expected)
    if not ok:
        return CompareResult(passed=False, reason=f"parameter names wrong: {msg}")

    for name, shape in [
        ("gate_proj", (d_ff, d_model)),
        ("up_proj", (d_ff, d_model)),
        ("down_proj", (d_model, d_ff)),
    ]:
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


# ---- Forward checks (with weight sync) ------------------------------------


def _run_forward(user_module, d_model: int, d_ff: int, shape: tuple, seed: int):
    user_mod = user_module.SwiGLUFFN(d_model=d_model, d_ff=d_ff)
    ref_mod = _REF.SwiGLUFFN(d_model=d_model, d_ff=d_ff)
    sync_weights(user_mod, ref_mod)
    user_mod.eval()
    ref_mod.eval()
    x = torch.randn(*shape, generator=_g(seed))
    return user_mod(x.clone()), ref_mod(x.clone())


TEST_CASES = [
    TestCase(
        name="init / 3 Linear(bias=False) with correct shapes",
        runner=_check_init,
        weight=2.0,
    ),
    TestCase(
        name="forward / 2D (B, d_model)",
        runner=lambda m: _run_forward(m, d_model=32, d_ff=64, shape=(4, 32), seed=10),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="forward / 3D (B, T, d_model)",
        runner=lambda m: _run_forward(
            m, d_model=16, d_ff=48, shape=(2, 6, 16), seed=20
        ),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="forward / LLaMA-ish ratio d_ff ≈ 2.67 * d_model",
        runner=lambda m: _run_forward(
            m, d_model=24, d_ff=64, shape=(3, 5, 24), seed=30
        ),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
]
