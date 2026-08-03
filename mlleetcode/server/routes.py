"""API routes."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..judge import JudgeReport, judge_submission
from ..registry import PROBLEMS_DIR, find_problem, list_problems
from .schemas import (
    CaseResultSchema,
    JudgeResultSchema,
    ProblemDetail,
    ProblemMeta,
    ProblemSolution,
    ProgressSchema,
    StatusEntry,
    SubmitRequest,
)

router = APIRouter(prefix="/api")

HISTORY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "workspace" / ".history.json"
)


def _load_history() -> dict:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return {}


def _save_history(data: dict) -> None:
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@router.get("/problems", response_model=list[ProblemMeta])
def get_problems(prefix: str | None = None):
    problems = list_problems(prefix=prefix or None)
    return [
        ProblemMeta(
            id=p.id,
            slug=p.slug,
            category=p.category,
            title=p.title,
            difficulty=p.difficulty,
            framework=p.framework,
            tags=p.tags,
        )
        for p in problems
    ]


@router.get("/problems/{problem_id:path}", response_model=ProblemDetail)
def get_problem(problem_id: str):
    try:
        p = find_problem(problem_id)
    except LookupError as e:
        raise HTTPException(404, str(e))
    readme = p.readme_path.read_text(encoding="utf-8") if p.readme_path.exists() else ""
    starter = (
        p.starter_path.read_text(encoding="utf-8") if p.starter_path.exists() else ""
    )
    return ProblemDetail(
        id=p.id,
        title=p.title,
        difficulty=p.difficulty,
        framework=p.framework,
        tags=p.tags,
        readme=readme,
        starter=starter,
    )


@router.get("/solution/{problem_id:path}", response_model=ProblemSolution)
def get_solution(problem_id: str):
    try:
        p = find_problem(problem_id)
    except LookupError as e:
        raise HTTPException(404, str(e))
    sol_md = (
        p.solution_md_path.read_text(encoding="utf-8")
        if p.solution_md_path.exists()
        else ""
    )
    sol_py = (
        p.solution_path.read_text(encoding="utf-8") if p.solution_path.exists() else ""
    )
    return ProblemSolution(id=p.id, solution_md=sol_md, solution_py=sol_py)


@router.post("/submit", response_model=JudgeResultSchema)
def submit(req: SubmitRequest):
    try:
        problem = find_problem(req.problem_id)
    except LookupError as e:
        raise HTTPException(404, str(e))

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".py",
        delete=False,
        dir=tempfile.gettempdir(),
    ) as f:
        f.write(req.code)
        tmp_path = Path(f.name)

    try:
        report = judge_submission(problem, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    _record_submission(problem.id, report)

    cases = []
    for c in report.cases:
        cases.append(
            CaseResultSchema(
                name=c.name,
                passed=c.passed,
                elapsed_ms=c.elapsed * 1000,
                weight=c.weight,
                reason=c.reason,
                expected_preview=c.compare.expected_preview if c.compare else "",
                actual_preview=c.compare.actual_preview if c.compare else "",
            )
        )
    return JudgeResultSchema(
        problem_id=problem.id,
        score=report.score,
        all_passed=report.all_passed,
        load_error=report.load_error,
        cases=cases,
    )


def _record_submission(problem_id: str, report: JudgeReport) -> None:
    history = _load_history()
    entry = history.setdefault(
        problem_id, {"best_score": 0.0, "attempts": 0, "last_attempt": ""}
    )
    entry["attempts"] += 1
    entry["last_attempt"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if report.score > entry["best_score"]:
        entry["best_score"] = float(report.score)
    _save_history(history)


@router.get("/status", response_model=ProgressSchema)
def get_status():
    history = _load_history()
    all_problems = list_problems()
    entries = []
    for p in all_problems:
        h = history.get(p.id)
        if h:
            entries.append(
                StatusEntry(
                    problem_id=p.id,
                    best_score=h["best_score"],
                    attempts=h["attempts"],
                    last_attempt=h["last_attempt"],
                )
            )
        else:
            entries.append(
                StatusEntry(
                    problem_id=p.id,
                    best_score=0.0,
                    attempts=0,
                    last_attempt="",
                )
            )
    attempted = sum(1 for e in entries if e.attempts > 0)
    perfect = sum(1 for e in entries if e.best_score >= 100.0)
    return ProgressSchema(
        total=len(all_problems),
        attempted=attempted,
        perfect=perfect,
        entries=entries,
    )


@router.delete("/status")
def reset_status():
    """Clear all recorded progress (history)."""
    _save_history({})
    return {"ok": True}
