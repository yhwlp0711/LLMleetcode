"""Test cases for pytorch.llm.positional.rope."""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult, compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_rope")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


# ---- build_rope_cache ------------------------------------------------------


def _run_cache(user_module, seq_len: int, head_dim: int, base: float):
    user_cos, user_sin = user_module.build_rope_cache(seq_len, head_dim, base=base)
    ref_cos, ref_sin = _REF.build_rope_cache(seq_len, head_dim, base=base)
    # combine into one tensor so we get a single comparison
    return torch.stack([user_cos, user_sin]), torch.stack([ref_cos, ref_sin])


# ---- apply_rope ------------------------------------------------------------


def _run_apply(user_module, B: int, H: int, T: int, D: int, seed: int):
    x = torch.randn(B, H, T, D, generator=_g(seed))
    cos, sin = _REF.build_rope_cache(T, D)
    actual = user_module.apply_rope(x.clone(), cos.clone(), sin.clone())
    expected = _REF.apply_rope(x.clone(), cos.clone(), sin.clone())
    return actual, expected


# ---- Sanity property: position-0 token is unchanged -----------------------


def _run_position_zero_identity(user_module) -> CompareResult:
    B, H, T, D = 2, 4, 8, 16
    x = torch.randn(B, H, T, D, generator=_g(99))
    cos, sin = _REF.build_rope_cache(T, D)
    out = user_module.apply_rope(x.clone(), cos.clone(), sin.clone())
    # At position 0, angle = 0 so cos=1, sin=0 -> out[..., 0, :] should equal x[..., 0, :]
    return compare_numeric(out[:, :, 0, :], x[:, :, 0, :], atol=1e-6, rtol=1e-6)


TEST_CASES = [
    TestCase(
        name="build_rope_cache / short",
        runner=lambda m: _run_cache(m, seq_len=8, head_dim=16, base=10000.0),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="build_rope_cache / longer + smaller base",
        runner=lambda m: _run_cache(m, seq_len=64, head_dim=32, base=1000.0),
        weight=1.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="apply_rope / basic",
        runner=lambda m: _run_apply(m, B=1, H=4, T=8, D=16, seed=1),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="apply_rope / multi-batch larger T",
        runner=lambda m: _run_apply(m, B=2, H=8, T=32, D=64, seed=2),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="property / position 0 leaves x unchanged",
        runner=_run_position_zero_identity,
        weight=1.0,
    ),
]
