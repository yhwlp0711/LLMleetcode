"""Test cases for numpy.basics.sliding_window."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_slidewin")


def _rng(seed: int):
    return np.random.default_rng(seed)


def _fx_swin_basic():
    return np.arange(10, dtype=np.float64), 3, 1


def _fx_swin_strided():
    return _rng(1).standard_normal(20), 4, 2


def _fx_swin_stride_eq_window():
    return _rng(2).standard_normal(16), 4, 4


def _fx_ma_short():
    return _rng(3).standard_normal(30), 5


def _fx_ma_window_eq_len():
    return _rng(4).standard_normal(8), 8


def _fx_conv_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k = np.array([1.0, 0.0, -1.0])
    return x, k


def _fx_conv_random():
    return _rng(6).standard_normal(50), _rng(7).standard_normal(7)


TEST_CASES = [
    TestCase(
        name="sliding_window_1d / basic",
        runner=lambda m: (
            m.sliding_window_1d(*_fx_swin_basic()),
            _REF.sliding_window_1d(*_fx_swin_basic()),
        ),
        weight=1.0,
    ),
    TestCase(
        name="sliding_window_1d / stride=2",
        runner=lambda m: (
            m.sliding_window_1d(*_fx_swin_strided()),
            _REF.sliding_window_1d(*_fx_swin_strided()),
        ),
        weight=1.0,
    ),
    TestCase(
        name="sliding_window_1d / non-overlapping",
        runner=lambda m: (
            m.sliding_window_1d(*_fx_swin_stride_eq_window()),
            _REF.sliding_window_1d(*_fx_swin_stride_eq_window()),
        ),
        weight=1.0,
    ),
    TestCase(
        name="moving_average / short window",
        runner=lambda m: (
            m.moving_average(*_fx_ma_short()),
            _REF.moving_average(*_fx_ma_short()),
        ),
        weight=1.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="moving_average / window=len",
        runner=lambda m: (
            m.moving_average(*_fx_ma_window_eq_len()),
            _REF.moving_average(*_fx_ma_window_eq_len()),
        ),
        weight=1.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="conv1d_valid / known answer",
        runner=lambda m: (
            m.conv1d_valid(*_fx_conv_basic()),
            _REF.conv1d_valid(*_fx_conv_basic()),
        ),
        weight=1.0,
        atol=1e-10,
        rtol=1e-10,
    ),
    TestCase(
        name="conv1d_valid / random",
        runner=lambda m: (
            m.conv1d_valid(*_fx_conv_random()),
            _REF.conv1d_valid(*_fx_conv_random()),
        ),
        weight=2.0,
        atol=1e-10,
        rtol=1e-10,
    ),
]
