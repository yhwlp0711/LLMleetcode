"""Command-line interface for mlleetcode."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

from . import __version__
from .judge import judge_submission
from .registry import PROBLEMS_DIR, find_problem, list_problems
from .report import (
    console,
    render_judge_report,
    render_problem_detail,
    render_problem_list,
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"


@click.group(help="LeetCode-style hand-coding judge for ML/LLM interviews.")
@click.version_option(__version__, prog_name="mlleetcode")
def main() -> None:
    pass


@main.command("list", help="List all problems, optionally filtered by a path prefix.")
@click.argument("prefix", required=False)
def list_cmd(prefix: str | None) -> None:
    render_problem_list(list_problems(prefix=prefix), filter_prefix=prefix)


@main.command("show", help="Show problem description.")
@click.argument("identifier")
def show_cmd(identifier: str) -> None:
    try:
        problem = find_problem(identifier)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    render_problem_detail(problem)


@main.command("solution", help="Show the reference solution and walkthrough.")
@click.argument("identifier")
def solution_cmd(identifier: str) -> None:
    try:
        problem = find_problem(identifier)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    from .report import render_problem_solution

    render_problem_solution(problem)


@main.command("start", help="Copy starter code into workspace/ for editing.")
@click.argument("identifier")
@click.option("--force", is_flag=True, help="Overwrite existing workspace file.")
def start_cmd(identifier: str, force: bool) -> None:
    try:
        problem = find_problem(identifier)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    WORKSPACE_DIR.mkdir(exist_ok=True)
    dest = WORKSPACE_DIR / problem.workspace_filename
    if dest.exists() and not force:
        console.print(f"[yellow]Already exists:[/yellow] {dest}")
        console.print("Use [bold]--force[/bold] to overwrite, or edit in place.")
    else:
        shutil.copyfile(problem.starter_path, dest)
        console.print(f"[green]Created:[/green] {dest}")
    console.print(f"[dim]Submit with:[/dim] mlleetcode submit {dest}")


def _infer_id_from_filename(submission: Path) -> str:
    # workspace files use __ as path separator; turn it back into dotted id
    return submission.stem.replace("__", ".")


@main.command("submit", help="Judge a submission file.")
@click.argument(
    "submission", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--problem",
    "-p",
    "problem_id",
    default=None,
    help="Problem id (inferred from filename if omitted).",
)
@click.option("--seed", default=42, show_default=True, type=int)
def submit_cmd(submission: Path, problem_id: str | None, seed: int) -> None:
    identifier = problem_id or _infer_id_from_filename(submission)
    try:
        problem = find_problem(identifier)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[dim]Hint: pass --problem <id> explicitly.[/dim]")
        sys.exit(1)

    report = judge_submission(problem, submission, seed=seed)
    render_judge_report(report, problem)
    sys.exit(0 if report.all_passed else 1)


@main.command(
    "verify", help="Self-check: judge the reference solution; must pass 100%."
)
@click.argument("identifier", required=False)
def verify_cmd(identifier: str | None) -> None:
    if identifier:
        try:
            problems = [find_problem(identifier)]
        except LookupError:
            # treat as a prefix filter (e.g. 'pytorch' to verify all pytorch problems)
            problems = list_problems(prefix=identifier)
            if not problems:
                console.print(f"[red]No problems found for '{identifier}'[/red]")
                sys.exit(1)
    else:
        problems = list_problems()

    all_ok = True
    runtime_warnings: list[tuple[str, float]] = []
    for problem in problems:
        if not problem.solution_path.exists():
            console.print(f"[yellow]Skip {problem.id}: no solution.py[/yellow]")
            continue
        report = judge_submission(problem, problem.solution_path)
        render_judge_report(report, problem)
        if not report.all_passed:
            all_ok = False
        total_s = sum(c.elapsed for c in report.cases)
        if total_s > 10.0:
            console.print(
                f"[bold red]⏱  {problem.id} total runtime {total_s:.1f}s — "
                "TOO SLOW (>10s). Shrink fixtures.[/bold red]"
            )
            runtime_warnings.append((problem.id, total_s))
        elif total_s > 2.0:
            console.print(
                f"[yellow]⏱  {problem.id} total runtime {total_s:.1f}s — "
                "above 2s budget; consider shrinking fixtures.[/yellow]"
            )
            runtime_warnings.append((problem.id, total_s))

    if runtime_warnings:
        console.rule(
            "[bold yellow]Runtime budget warnings[/bold yellow]", style="yellow"
        )
        for pid, t in runtime_warnings:
            color = "red" if t > 10.0 else "yellow"
            console.print(f"  [{color}]• {pid}: {t:.2f}s[/{color}]")
    sys.exit(0 if all_ok else 1)


@main.command("ui", help="Launch the local web UI.")
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--no-open", is_flag=True, help="Don't auto-open the browser.")
def ui_cmd(port: int, no_open: bool) -> None:
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn not installed.[/red] "
            'Run: [bold]pip install -e ".[web]"[/bold]'
        )
        sys.exit(1)

    from pathlib import Path

    dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if not dist.exists():
        console.print(
            "[yellow]Warning:[/yellow] web/dist/ not found. "
            "The UI won't serve frontend files.\n"
            "If developing: run [bold]cd web && npm run build[/bold] first."
        )

    url = f"http://localhost:{port}"
    console.print(f"[bold cyan]Starting mlleetcode UI at {url}[/bold cyan]")
    if not no_open:
        import webbrowser

        webbrowser.open(url)
    uvicorn.run(
        "mlleetcode.server.app:app", host="0.0.0.0", port=port, log_level="info"
    )


if __name__ == "__main__":
    main()
