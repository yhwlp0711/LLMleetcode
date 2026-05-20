"""Test cases for pytorch.nn.rmsnorm."""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult
from mlleetcode.utils.sandbox import load_module_from_path
from mlleetcode.utils.stats import check_shape
from mlleetcode.utils.weights import assert_param_names, sync_weights

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_rmsnorm")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _check_init(user_module) -> CompareResult:
    D = 16
    mod = user_module.RMSNorm(normalized_dim=D)
    ok, msg = assert_param_names(mod, ["weight"])
    if not ok:
        return CompareResult(passed=False, reason=f"parameters wrong: {msg}")
    s = check_shape(mod.weight, (D,), name="weight")
    if not s.passed:
        return s
    if not torch.allclose(mod.weight.detach(), torch.ones(D)):
        return CompareResult(passed=False, reason="weight not initialized to ones")
    return CompareResult(passed=True)


def _run_forward(user_module, dim: int, shape: tuple, seed: int, randomize: bool):
    user_mod = user_module.RMSNorm(normalized_dim=dim)
    ref_mod = _REF.RMSNorm(normalized_dim=dim)
    if randomize:
        with torch.no_grad():
            ref_mod.weight.copy_(torch.randn(dim, generator=_g(seed + 1000)))
    sync_weights(user_mod, ref_mod)
    x = torch.randn(*shape, generator=_g(seed))
    return user_mod(x.clone()), ref_mod(x.clone())


TEST_CASES = [
    TestCase(name="init / weight ones", runner=_check_init, weight=2.0),
    TestCase(
        name="forward / 2D, default weight",
        runner=lambda m: _run_forward(m, 16, (4, 16), seed=10, randomize=False),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="forward / 3D, default weight",
        runner=lambda m: _run_forward(m, 8, (2, 6, 8), seed=20, randomize=False),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="forward / 3D, randomized weight",
        runner=lambda m: _run_forward(m, 12, (3, 5, 12), seed=30, randomize=True),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
]
