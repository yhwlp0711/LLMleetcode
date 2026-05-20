"""Test cases for numpy.basics.broadcasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_broadcast")


def _rng(seed: int):
    return np.random.default_rng(seed)


def _fx_outer():
    return _rng(0).standard_normal(7), _rng(1).standard_normal(5)


def _fx_pairwise():
    return _rng(2).standard_normal(6)


def _fx_norm():
    return _rng(3).standard_normal((20, 4))


def _fx_row_scale():
    rng = _rng(4)
    return rng.standard_normal((6, 3)), rng.standard_normal(6)


TEST_CASES = [
    TestCase(
        name="outer_sum",
        runner=lambda m: (m.outer_sum(*_fx_outer()), _REF.outer_sum(*_fx_outer())),
        weight=1.0,
        atol=1e-12,
        rtol=1e-12,
    ),
    TestCase(
        name="pairwise_difference",
        runner=lambda m: (
            m.pairwise_difference(_fx_pairwise()),
            _REF.pairwise_difference(_fx_pairwise()),
        ),
        weight=1.0,
        atol=1e-12,
        rtol=1e-12,
    ),
    TestCase(
        name="normalize_columns",
        runner=lambda m: (
            m.normalize_columns(_fx_norm()),
            _REF.normalize_columns(_fx_norm()),
        ),
        weight=2.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="apply_per_row_scale",
        runner=lambda m: (
            m.apply_per_row_scale(*_fx_row_scale()),
            _REF.apply_per_row_scale(*_fx_row_scale()),
        ),
        weight=1.0,
        atol=1e-12,
        rtol=1e-12,
    ),
]
