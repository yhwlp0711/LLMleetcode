"""Test cases for Tensor Ops Warmup.

Each helper gets its own TestCase(s). Expected outputs come from the reference
solution, so this file only declares the input fixtures.
"""
from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_tensor_ops")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


# ---- fixtures (built deterministically; not affected by user code) ----------

def _fx_flatten():
    return (
        torch.randn(3, 4, generator=_g(0)),
        torch.randn(2, 5, 2, generator=_g(1)),
    )


def _fx_softmax_small():
    return torch.randn(4, 6, generator=_g(2)) * 5.0


def _fx_softmax_extreme():
    # Includes very large/small values; tests numerical stability.
    x = torch.randn(3, 5, generator=_g(3)) * 1.0
    x[0, 2] = 1e3
    x[1, :] = -1e3
    return x


def _fx_pairwise():
    return torch.randn(7, 4, generator=_g(4)), torch.randn(5, 4, generator=_g(5))


def _fx_masked_mean():
    x = torch.randn(2, 6, 3, generator=_g(6))
    mask = torch.tensor([
        [True, True, False, True, False, False],
        [True, False, False, False, False, False],
    ])
    return x, mask


def _fx_topk():
    return torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]), 3


def _fx_topk_with_ties():
    # All equal — should return indices 0..k-1 in order.
    return torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]), 3


# ---- TestCase list ----------------------------------------------------------

TEST_CASES = [
    TestCase(
        name="flatten_and_concat",
        runner=lambda m: (
            m.flatten_and_concat(*_fx_flatten()),
            _REF.flatten_and_concat(*_fx_flatten()),
        ),
        weight=1.0,
    ),

    TestCase(
        name="row_softmax / small inputs",
        runner=lambda m: (m.row_softmax(_fx_softmax_small()), _REF.row_softmax(_fx_softmax_small())),
        weight=1.0,
    ),
    TestCase(
        name="row_softmax / numerical stability",
        runner=lambda m: (m.row_softmax(_fx_softmax_extreme()), _REF.row_softmax(_fx_softmax_extreme())),
        weight=1.0,
        atol=1e-6, rtol=1e-6,
    ),

    TestCase(
        name="pairwise_squared_distance",
        runner=lambda m: (
            m.pairwise_squared_distance(*_fx_pairwise()),
            _REF.pairwise_squared_distance(*_fx_pairwise()),
        ),
        weight=2.0,
        atol=1e-5, rtol=1e-5,
    ),

    TestCase(
        name="masked_mean",
        runner=lambda m: (m.masked_mean(*_fx_masked_mean()), _REF.masked_mean(*_fx_masked_mean())),
        weight=1.0,
    ),

    TestCase(
        name="top_k_indices / typical",
        runner=lambda m: (m.top_k_indices(*_fx_topk()), _REF.top_k_indices(*_fx_topk())),
        weight=1.0,
    ),
    TestCase(
        name="top_k_indices / all ties",
        runner=lambda m: (m.top_k_indices(*_fx_topk_with_ties()), _REF.top_k_indices(*_fx_topk_with_ties())),
        weight=1.0,
    ),
]
