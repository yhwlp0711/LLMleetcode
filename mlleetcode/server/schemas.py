"""Pydantic response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class ProblemMeta(BaseModel):
    id: str
    slug: str
    category: str
    title: str
    difficulty: str
    framework: str
    tags: list[str]


class ProblemDetail(BaseModel):
    id: str
    title: str
    difficulty: str
    framework: str
    tags: list[str]
    readme: str
    starter: str


class ProblemSolution(BaseModel):
    id: str
    solution_md: str
    solution_py: str


class CaseResultSchema(BaseModel):
    name: str
    passed: bool
    elapsed_ms: float
    weight: float
    reason: str
    expected_preview: str
    actual_preview: str


class JudgeResultSchema(BaseModel):
    problem_id: str
    score: float
    all_passed: bool
    load_error: str
    cases: list[CaseResultSchema]


class SubmitRequest(BaseModel):
    problem_id: str
    code: str


class StatusEntry(BaseModel):
    problem_id: str
    best_score: float
    attempts: int
    last_attempt: str


class ProgressSchema(BaseModel):
    total: int
    attempted: int
    perfect: int
    entries: list[StatusEntry]
