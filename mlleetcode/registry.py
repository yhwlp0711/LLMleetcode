"""Discover and load problems from the problems/ tree.

Problems are identified by a dotted slug-path derived from their directory under
problems/. For example:

    problems/numpy/ml/linear_regression/   ->  id = "numpy.ml.linear_regression"
    problems/pytorch/llm/mha/              ->  id = "pytorch.llm.mha"

A problem directory is any directory that contains a meta.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"

_WORKSPACE_SEP = "__"


@dataclass
class Problem:
    id: str  # dotted slug-path, e.g. "pytorch.llm.mha"
    slug: str  # leaf dir name, e.g. "mha"
    category: str  # parent path joined by "." e.g. "pytorch.llm"
    title: str
    difficulty: str
    framework: str  # "numpy" | "pytorch" | "mixed" | ...
    tags: list[str] = field(default_factory=list)
    timeout: float = 10.0
    entrypoint: str = "solve"
    dir: Path = field(default=Path("."))

    @property
    def readme_path(self) -> Path:
        return self.dir / "README.md"

    @property
    def solution_md_path(self) -> Path:
        return self.dir / "solution.md"

    @property
    def starter_path(self) -> Path:
        return self.dir / "starter.py"

    @property
    def solution_path(self) -> Path:
        return self.dir / "solution.py"

    @property
    def test_cases_path(self) -> Path:
        return self.dir / "test_cases.py"

    @property
    def workspace_filename(self) -> str:
        """The flat workspace filename for this problem (path joined by __)."""
        return self.id.replace(".", _WORKSPACE_SEP) + ".py"


def _make_problem(problem_dir: Path, root: Path) -> Problem:
    meta_path = problem_dir / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) or {}

    rel_parts = problem_dir.relative_to(root).parts
    slug = rel_parts[-1]
    category = ".".join(rel_parts[:-1])
    pid = ".".join(rel_parts)

    return Problem(
        id=pid,
        slug=slug,
        category=category,
        title=meta.get("title", slug),
        difficulty=meta.get("difficulty", "unknown"),
        framework=meta.get("framework", "unknown"),
        tags=list(meta.get("tags", [])),
        timeout=float(meta.get("timeout", 10.0)),
        entrypoint=meta.get("entrypoint", "solve"),
        dir=problem_dir,
    )


def list_problems(
    problems_dir: Path = PROBLEMS_DIR,
    prefix: str | None = None,
) -> list[Problem]:
    """List problems, optionally filtered by a dotted-path prefix (e.g. 'pytorch.llm')."""
    if not problems_dir.exists():
        return []
    problems: list[Problem] = []
    for meta_path in sorted(problems_dir.rglob("meta.yaml")):
        problem_dir = meta_path.parent
        problems.append(_make_problem(problem_dir, problems_dir))
    if prefix:
        norm = prefix.strip(".").replace("/", ".")
        problems = [p for p in problems if p.id == norm or p.id.startswith(norm + ".")]
    return problems


def _normalize(identifier: str) -> str:
    return identifier.strip().strip(".").replace("/", ".").replace(_WORKSPACE_SEP, ".")


def find_problem(identifier: str, problems_dir: Path = PROBLEMS_DIR) -> Problem:
    """Resolve an identifier to a Problem with several matching strategies.

    Tried in order:
      1. exact id match (numpy.ml.linear_regression)
      2. exact slug match (linear_regression) — must be unique
      3. suffix match on id (e.g. 'llm.mha' matches 'pytorch.llm.mha') — must be unique
      4. substring match on id — must be unique
      5. per-segment prefix match (e.g. 'numpy.ml.lr' matches 'numpy.ml.linear_regression')

    Raises LookupError when ambiguous or not found.
    """
    norm = _normalize(identifier)
    problems = list_problems(problems_dir)

    # 1. exact id
    exact = [p for p in problems if p.id == norm]
    if exact:
        return exact[0]

    # 2. exact slug (leaf)
    by_slug = [p for p in problems if p.slug == norm]
    if len(by_slug) == 1:
        return by_slug[0]
    if len(by_slug) > 1:
        raise LookupError(
            f"Slug '{norm}' is ambiguous: {[p.id for p in by_slug]}. "
            "Use the full dotted id."
        )

    # 3. suffix match
    suffix = [p for p in problems if p.id.endswith("." + norm)]
    if len(suffix) == 1:
        return suffix[0]
    if len(suffix) > 1:
        raise LookupError(f"Suffix '{norm}' is ambiguous: {[p.id for p in suffix]}.")

    # 4. per-segment prefix match (each segment of `norm` must be a prefix of
    # the corresponding tail-aligned segment of p.id)
    needle = norm.split(".")
    prefix_matches = []
    for p in problems:
        parts = p.id.split(".")
        if len(parts) < len(needle):
            continue
        tail = parts[-len(needle) :]
        if all(have.startswith(want) for want, have in zip(needle, tail)):
            prefix_matches.append(p)
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise LookupError(
            f"Identifier '{norm}' is ambiguous: {[p.id for p in prefix_matches]}."
        )

    # 5. substring fallback
    substring = [p for p in problems if norm in p.id]
    if len(substring) == 1:
        return substring[0]
    if len(substring) > 1:
        raise LookupError(
            f"Identifier '{norm}' is ambiguous: {[p.id for p in substring]}."
        )

    # 6. acronym match on the leaf slug (e.g. 'sdpa' -> 'scaled_dot_product_attention')
    acronym_matches = []
    for p in problems:
        leaf_parts = p.slug.split("_")
        acronym = "".join(part[0] for part in leaf_parts if part)
        if acronym == norm:
            acronym_matches.append(p)
    if len(acronym_matches) == 1:
        return acronym_matches[0]
    if len(acronym_matches) > 1:
        raise LookupError(
            f"Acronym '{norm}' is ambiguous: {[p.id for p in acronym_matches]}."
        )

    raise LookupError(f"No problem matches '{identifier}'")
