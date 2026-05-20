"""Test cases for pytorch.nn.layernorm.

Demonstrates Pattern B (module problem):
  - Init checks: parameter shapes + initial values (no need for state_dict injection here)
  - Forward checks: weights synced from reference, then numeric comparison
"""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult, compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path
from mlleetcode.utils.stats import check_shape
from mlleetcode.utils.weights import assert_param_names, sync_weights

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_layernorm")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


# ---- Init checks -----------------------------------------------------------


def _check_init(user_module) -> CompareResult:
    D = 16
    mod = user_module.LayerNorm(normalized_dim=D)
    names_ok, msg = assert_param_names(mod, ["weight", "bias"])
    if not names_ok:
        return CompareResult(passed=False, reason=f"parameter names wrong: {msg}")
    shape_w = check_shape(mod.weight, (D,), name="weight")
    if not shape_w.passed:
        return shape_w
    shape_b = check_shape(mod.bias, (D,), name="bias")
    if not shape_b.passed:
        return shape_b
    # Check initial values
    if not torch.allclose(mod.weight.detach(), torch.ones(D)):
        return CompareResult(passed=False, reason="weight not initialized to ones")
    if not torch.allclose(mod.bias.detach(), torch.zeros(D)):
        return CompareResult(passed=False, reason="bias not initialized to zeros")
    return CompareResult(passed=True)


# ---- Forward checks (with weight sync) -------------------------------------


def _run_forward(
    user_module, dim: int, shape: tuple, seed: int, randomize_weights: bool
):
    user_mod = user_module.LayerNorm(normalized_dim=dim)
    ref_mod = _REF.LayerNorm(normalized_dim=dim)
    if randomize_weights:
        # Replace reference weights with random non-identity values, then sync.
        g = _g(seed + 1000)
        with torch.no_grad():
            ref_mod.weight.copy_(torch.randn(dim, generator=g))
            ref_mod.bias.copy_(torch.randn(dim, generator=g))
    sync_weights(user_mod, ref_mod)

    x = torch.randn(*shape, generator=_g(seed))
    return user_mod(x.clone()), ref_mod(x.clone())


TEST_CASES = [
    TestCase(name="init / shapes + initial values", runner=_check_init, weight=2.0),
    TestCase(
        name="forward / 2D (B, D), default weights",
        runner=lambda m: _run_forward(
            m, dim=16, shape=(4, 16), seed=10, randomize_weights=False
        ),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="forward / 3D (B, T, D), default weights",
        runner=lambda m: _run_forward(
            m, dim=8, shape=(2, 6, 8), seed=20, randomize_weights=False
        ),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="forward / 3D, randomized weight & bias",
        runner=lambda m: _run_forward(
            m, dim=12, shape=(3, 5, 12), seed=30, randomize_weights=True
        ),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
]
