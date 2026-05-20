"""Judging engine.

A TestCase has a `runner(user_module)` that returns one of:
  - (actual, expected): the engine compares them with compare_numeric
  - CompareResult: the runner did its own judging (e.g. shape/distribution check)
  - bool: pass/fail with no further detail (use sparingly)

This lets a single TestCase abstraction cover numeric tests, init-distribution
tests, shape checks, etc.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .registry import Problem
from .utils.compare import CompareResult, compare_numeric
from .utils.device import device_tolerance, get_device
from .utils.sandbox import (
    SubmissionLoadError,
    TimeoutError_,
    load_module_from_path,
    time_limit,
)
from .utils.seed import set_seed


@dataclass
class CaseResult:
    name: str
    passed: bool
    elapsed: float
    weight: float = 1.0
    reason: str = ""
    compare: CompareResult | None = None
    traceback_str: str = ""


@dataclass
class JudgeReport:
    problem_id: str
    submission_path: str
    cases: list[CaseResult] = field(default_factory=list)
    load_error: str = ""

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.cases) or 1.0

    @property
    def earned(self) -> float:
        return sum(c.weight for c in self.cases if c.passed)

    @property
    def score(self) -> float:
        return 100.0 * self.earned / self.total_weight

    @property
    def all_passed(self) -> bool:
        return bool(self.cases) and all(c.passed for c in self.cases)


class _NoCollect:
    # Prevent pytest from auto-collecting classes named Test* defined here.
    __test__ = False


@dataclass
class TestCase(_NoCollect):
    name: str
    runner: Callable[[Any], Any]   # see module docstring for accepted return types
    weight: float = 1.0
    atol: float = 1e-5
    rtol: float = 1e-4
    description: str = ""


# --- Engine -----------------------------------------------------------------

def judge_submission(problem: Problem, submission_path: Path, seed: int = 42) -> JudgeReport:
    report = JudgeReport(problem_id=problem.id, submission_path=str(submission_path))

    try:
        user_module = load_module_from_path(submission_path)
    except SubmissionLoadError as exc:
        report.load_error = str(exc)
        return report

    try:
        tc_module = load_module_from_path(problem.test_cases_path)
    except SubmissionLoadError as exc:
        report.load_error = f"Failed to load problem test cases: {exc}"
        return report

    cases: list[TestCase] = getattr(tc_module, "TEST_CASES", [])
    if not cases:
        report.load_error = f"No TEST_CASES defined in {problem.test_cases_path}"
        return report

    device = get_device()
    for case in cases:
        report.cases.append(_run_case(case, user_module, problem.timeout, device, seed))

    return report


def _run_case(case: TestCase, user_module: Any, timeout: float, device, seed: int) -> CaseResult:
    set_seed(seed)
    atol, rtol = device_tolerance(device, case.atol, case.rtol)
    start = time.perf_counter()
    try:
        with time_limit(timeout):
            out = case.runner(user_module)
        elapsed = time.perf_counter() - start
    except TimeoutError_ as exc:
        return CaseResult(
            name=case.name, passed=False, elapsed=time.perf_counter() - start,
            weight=case.weight, reason=str(exc),
        )
    except Exception as exc:
        return CaseResult(
            name=case.name, passed=False, elapsed=time.perf_counter() - start,
            weight=case.weight, reason=f"{type(exc).__name__}: {exc}",
            traceback_str=traceback.format_exc(),
        )

    return _interpret_runner_output(case, out, elapsed, atol, rtol)


def _interpret_runner_output(
    case: TestCase, out: Any, elapsed: float, atol: float, rtol: float,
) -> CaseResult:
    # Form 1: runner did its own judging.
    if isinstance(out, CompareResult):
        return CaseResult(
            name=case.name, passed=out.passed, elapsed=elapsed,
            weight=case.weight, reason=out.reason, compare=out,
        )

    # Form 2: simple bool.
    if isinstance(out, bool):
        return CaseResult(
            name=case.name, passed=out, elapsed=elapsed, weight=case.weight,
            reason="" if out else "runner returned False",
        )

    # Form 3: (actual, expected) -> numeric comparison.
    if isinstance(out, tuple) and len(out) == 2:
        actual, expected = out
        cmp = compare_numeric(actual, expected, atol=atol, rtol=rtol)
        return CaseResult(
            name=case.name, passed=cmp.passed, elapsed=elapsed,
            weight=case.weight, reason=cmp.reason, compare=cmp,
        )

    return CaseResult(
        name=case.name, passed=False, elapsed=elapsed, weight=case.weight,
        reason=(
            f"runner returned unsupported type {type(out).__name__}; "
            "expected (actual, expected) tuple, CompareResult, or bool."
        ),
    )
