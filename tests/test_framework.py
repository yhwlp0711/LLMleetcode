"""Unit tests for the judge engine and utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlleetcode.judge import judge_submission
from mlleetcode.registry import find_problem, list_problems
from mlleetcode.utils.compare import compare_numeric


# --- compare_numeric --------------------------------------------------------


def test_compare_passes_on_equal_arrays():
    a = np.array([1.0, 2.0, 3.0])
    res = compare_numeric(a, a.copy())
    assert res.passed
    assert res.max_abs_diff == 0.0


def test_compare_within_tolerance():
    a = np.array([1.0, 2.0])
    b = a + 1e-7
    assert compare_numeric(a, b, atol=1e-6, rtol=1e-6).passed


def test_compare_fails_on_shape_mismatch():
    res = compare_numeric(np.zeros(3), np.zeros(4))
    assert not res.passed
    assert "shape mismatch" in res.reason


def test_compare_fails_beyond_tolerance():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 1.1])
    res = compare_numeric(a, b, atol=1e-6, rtol=1e-6)
    assert not res.passed
    assert res.max_abs_diff is not None and res.max_abs_diff > 0.09


def test_compare_accepts_scalars_and_lists():
    assert compare_numeric(1.0, 1.0).passed
    assert compare_numeric([1.0, 2.0], [1.0, 2.0]).passed


# --- registry ---------------------------------------------------------------


def test_registry_lists_example_problem():
    ids = [p.id for p in list_problems()]
    assert "numpy.ml.linear_regression" in ids


def test_registry_find_strategies():
    full = find_problem("numpy.ml.linear_regression")
    by_segment_prefix = find_problem("numpy.ml.lin")
    by_substring = find_problem("numpy.ml.linear_regress")  # substring of full id
    assert full.id == by_segment_prefix.id == by_substring.id


def test_registry_ambiguous_slug_raises():
    # Both numpy and pytorch ship `linear_regression`; bare slug must be ambiguous.
    with pytest.raises(LookupError):
        find_problem("linear_regression")


def test_registry_acronym_match():
    # 'sdpa' should match 'scaled_dot_product_attention'
    p = find_problem("sdpa")
    assert p.id == "pytorch.llm.scaled_dot_product_attention"


def test_registry_prefix_filter():
    np_only = list_problems(prefix="numpy")
    assert all(p.id.startswith("numpy.") for p in np_only)
    assert len(np_only) >= 1

    none = list_problems(prefix="nonexistent")
    assert none == []


def test_registry_raises_on_unknown():
    with pytest.raises(LookupError):
        find_problem("does_not_exist_zzz")


def test_workspace_filename_uses_double_underscore():
    p = find_problem("numpy.ml.linear_regression")
    assert p.workspace_filename == "numpy__ml__linear_regression.py"


# --- judge end-to-end -------------------------------------------------------


def test_solution_passes_all_cases():
    problem = find_problem("numpy.ml.linear_regression")
    report = judge_submission(problem, problem.solution_path)
    assert report.load_error == ""
    assert report.all_passed, [
        (c.name, c.reason, getattr(c.compare, "max_abs_diff", None))
        for c in report.cases
        if not c.passed
    ]
    assert report.score == 100.0


def test_buggy_submission_fails(tmp_path: Path):
    buggy = tmp_path / "buggy.py"
    buggy.write_text(
        "import numpy as np\n"
        "def fit_predict(X_train, y_train, X_test, *, lr, epochs):\n"
        "    N, D = X_train.shape\n"
        "    w = np.zeros(D)\n"
        "    b = 0.0\n"
        "    for _ in range(epochs):\n"
        "        y_hat = X_train @ w + b\n"
        "        err = y_hat - y_train\n"
        "        w -= lr * (X_train.T @ err)\n"  # missing 2/N
        "        b -= lr * err.sum()\n"
        "    return w, float(b), X_test @ w + b\n"
    )
    problem = find_problem("numpy.ml.linear_regression")
    report = judge_submission(problem, buggy)
    assert not report.all_passed
    assert report.score < 100.0


def test_load_error_reported(tmp_path: Path):
    bad = tmp_path / "syntax_error.py"
    bad.write_text("def fit_predict(:\n")
    problem = find_problem("numpy.ml.linear_regression")
    report = judge_submission(problem, bad)
    assert report.load_error
    assert report.cases == []


# --- judge: runner return-type variants -------------------------------------


def test_runner_returning_compare_result(tmp_path: Path):
    """Verify the engine handles a runner that returns a CompareResult directly."""
    from mlleetcode.judge import TestCase, _run_case
    from mlleetcode.utils.compare import CompareResult

    case = TestCase(
        name="custom",
        runner=lambda m: CompareResult(passed=True, reason=""),
    )
    res = _run_case(case, user_module=None, timeout=5.0, device=None, seed=0)
    assert res.passed


def test_runner_returning_bool():
    from mlleetcode.judge import TestCase, _run_case

    case_ok = TestCase(name="ok", runner=lambda m: True)
    case_bad = TestCase(name="bad", runner=lambda m: False)
    assert _run_case(case_ok, None, 5.0, None, 0).passed
    assert not _run_case(case_bad, None, 5.0, None, 0).passed


def test_runner_returning_tuple():
    from mlleetcode.judge import TestCase, _run_case

    case = TestCase(name="tup", runner=lambda m: (np.zeros(3), np.zeros(3)))
    assert _run_case(case, None, 5.0, None, 0).passed
