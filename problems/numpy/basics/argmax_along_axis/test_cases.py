"""Test cases for numpy.basics.argmax_along_axis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlleetcode.judge import TestCase
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_argmax_axis")


def _rng(seed: int):
    return np.random.default_rng(seed)


def _fx_2d_axis0():
    return _rng(0).standard_normal((5, 7)), 0


def _fx_2d_axis1():
    return _rng(1).standard_normal((5, 7)), 1


def _fx_3d_axis_neg1():
    return _rng(2).standard_normal((3, 4, 6)), -1


def _fx_3d_axis1():
    return _rng(3).standard_normal((2, 5, 4)), 1


def _fx_ties():
    # Lots of equal values to test tie-breaking behaviour (lower index wins).
    a = np.array([[1.0, 3.0, 2.0, 3.0, 3.0], [5.0, 5.0, 5.0, 1.0, 5.0]])
    return a, 1


TEST_CASES = [
    TestCase(
        name="2D / axis=0",
        runner=lambda m: (
            m.argmax_along_axis(*_fx_2d_axis0()),
            _REF.argmax_along_axis(*_fx_2d_axis0()),
        ),
        weight=1.0,
    ),
    TestCase(
        name="2D / axis=1",
        runner=lambda m: (
            m.argmax_along_axis(*_fx_2d_axis1()),
            _REF.argmax_along_axis(*_fx_2d_axis1()),
        ),
        weight=1.0,
    ),
    TestCase(
        name="3D / axis=-1",
        runner=lambda m: (
            m.argmax_along_axis(*_fx_3d_axis_neg1()),
            _REF.argmax_along_axis(*_fx_3d_axis_neg1()),
        ),
        weight=2.0,
    ),
    TestCase(
        name="3D / axis=1",
        runner=lambda m: (
            m.argmax_along_axis(*_fx_3d_axis1()),
            _REF.argmax_along_axis(*_fx_3d_axis1()),
        ),
        weight=2.0,
    ),
    TestCase(
        name="ties / lowest-index wins",
        runner=lambda m: (
            m.argmax_along_axis(*_fx_ties()),
            _REF.argmax_along_axis(*_fx_ties()),
        ),
        weight=1.0,
    ),
]
