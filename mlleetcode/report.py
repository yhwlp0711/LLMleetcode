"""Rich-based pretty printing for problems, cases and reports."""

from __future__ import annotations

from collections import defaultdict

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .judge import JudgeReport
from .registry import Problem

console = Console()


_DIFFICULTY_STYLE = {
    "easy": "green",
    "medium": "yellow",
    "hard": "red",
}
_FRAMEWORK_STYLE = {
    "numpy": "blue",
    "pytorch": "magenta",
    "mixed": "cyan",
}


def _difficulty_text(d: str) -> Text:
    return Text(d, style=_DIFFICULTY_STYLE.get(d.lower(), "white"))


def _framework_text(f: str) -> Text:
    return Text(f, style=_FRAMEWORK_STYLE.get(f.lower(), "white"))


def render_problem_list(
    problems: list[Problem], filter_prefix: str | None = None
) -> None:
    if not problems:
        msg = "No problems found."
        if filter_prefix:
            msg += f" (filter: {filter_prefix})"
        console.print(f"[yellow]{msg}[/yellow]")
        return

    title = "[bold cyan]ML LeetCode — Problem List[/bold cyan]"
    if filter_prefix:
        title += f"  [dim](filter: {filter_prefix})[/dim]"

    table = Table(
        title=title,
        title_justify="left",
        show_lines=False,
        header_style="bold dim",
        padding=(0, 1),
        expand=False,
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", no_wrap=True, overflow="ellipsis", max_width=40)
    table.add_column("Diff", no_wrap=True)
    table.add_column(
        "Tags", style="dim", no_wrap=True, overflow="ellipsis", max_width=40
    )

    by_category: dict[str, list[Problem]] = defaultdict(list)
    for p in problems:
        by_category[p.category].append(p)

    first = True
    for category in sorted(by_category):
        if not first:
            table.add_section()
        first = False
        for p in by_category[category]:
            table.add_row(
                p.id,
                p.title,
                _difficulty_text(p.difficulty),
                ", ".join(p.tags),
            )
    console.print(table)


def render_problem_detail(problem: Problem) -> None:
    header = Text(f"{problem.id}", style="bold cyan")
    title = Text(problem.title, style="bold")
    meta_line = Text.assemble(
        ("difficulty: ", "dim"),
        _difficulty_text(problem.difficulty),
        ("    framework: ", "dim"),
        _framework_text(problem.framework),
        ("    tags: ", "dim"),
        (", ".join(problem.tags), "magenta"),
        ("    timeout: ", "dim"),
        (f"{problem.timeout:.0f}s", "white"),
    )
    console.print(
        Panel.fit(
            Text.assemble(header, "\n", title, "\n", meta_line),
            border_style="cyan",
        )
    )
    if problem.readme_path.exists():
        console.print(Markdown(problem.readme_path.read_text()))
    else:
        console.print("[yellow]README.md not found for this problem.[/yellow]")


def render_problem_solution(problem: Problem) -> None:
    """Show solution.md (Chinese walkthrough) followed by solution.py source."""
    header = Text(f"{problem.id} — 参考解答", style="bold cyan")
    console.print(Panel.fit(header, border_style="cyan"))

    if problem.solution_md_path.exists():
        console.print(Markdown(problem.solution_md_path.read_text()))
    else:
        console.print("[yellow](本题尚未提供 solution.md 中文解析)[/yellow]")

    if problem.solution_path.exists():
        console.rule("[bold]solution.py[/bold]", style="dim")
        console.print(
            Syntax(
                problem.solution_path.read_text(),
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=False,
            )
        )
    else:
        console.print("[red]solution.py not found.[/red]")


def render_judge_report(report: JudgeReport, problem: Problem | None = None) -> None:
    title = problem.title if problem else report.problem_id
    console.rule(
        f"[bold cyan]Judging:[/bold cyan] {title}  [dim]({report.problem_id})[/dim]"
    )
    console.print(f"[dim]submission:[/dim] {report.submission_path}")

    if report.load_error:
        console.print(
            Panel(
                Text(report.load_error, style="red"),
                title="Load Error",
                border_style="red",
            )
        )
        return

    table = Table(show_lines=False, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Case", style="bold")
    table.add_column("Result", justify="center", width=8)
    table.add_column("Time", justify="right", style="dim", width=8)
    table.add_column("Weight", justify="right", style="dim", width=6)
    table.add_column("Detail")

    for i, c in enumerate(report.cases, 1):
        status = (
            Text("PASS", style="bold green")
            if c.passed
            else Text("FAIL", style="bold red")
        )
        table.add_row(
            str(i),
            c.name,
            status,
            f"{c.elapsed * 1000:.1f}ms",
            f"{c.weight:g}",
            _format_case_detail(c),
        )
    console.print(table)

    # Failure detail panels
    for i, c in enumerate(report.cases, 1):
        if c.passed:
            continue
        body_lines: list[str] = []
        if c.reason:
            body_lines.append(f"[red]reason:[/red] {c.reason}")
        if c.compare:
            if c.compare.max_abs_diff is not None:
                body_lines.append(
                    f"[dim]max_abs_diff:[/dim] {c.compare.max_abs_diff:.3e}    "
                    f"[dim]max_rel_diff:[/dim] {c.compare.max_rel_diff:.3e}"
                )
            if c.compare.expected_preview:
                body_lines.append(
                    f"[green]expected:[/green] {c.compare.expected_preview}"
                )
            if c.compare.actual_preview:
                body_lines.append(f"[red]actual:  [/red] {c.compare.actual_preview}")
        if c.traceback_str:
            body_lines.append("[dim]" + c.traceback_str.rstrip() + "[/dim]")
        console.print(
            Panel(
                "\n".join(body_lines) if body_lines else "(no details)",
                title=f"Case #{i} — {c.name}",
                border_style="red",
            )
        )

    # Score panel
    score = report.score
    if report.all_passed:
        color, verdict = "green", "ACCEPTED"
    elif report.earned == 0:
        color, verdict = "red", "REJECTED"
    else:
        color, verdict = "yellow", "PARTIAL"
    summary = Text.assemble(
        (f"{verdict}", f"bold {color}"),
        "    ",
        (f"score: {score:.1f}/100", "bold"),
        "    ",
        (
            f"passed: {sum(c.passed for c in report.cases)}/{len(report.cases)} cases",
            "dim",
        ),
        "    ",
        (f"weight: {report.earned:g}/{report.total_weight:g}", "dim"),
    )
    console.print(Panel.fit(summary, border_style=color))


def _format_case_detail(c) -> str:
    if c.passed:
        return "[green]ok[/green]"
    if c.reason and not c.compare:
        return f"[red]{c.reason}[/red]"
    if c.compare and c.compare.max_abs_diff is not None:
        return f"[red]max|Δ|={c.compare.max_abs_diff:.2e}[/red]"
    if c.compare and c.compare.reason:
        return f"[red]{c.compare.reason}[/red]"
    return "[red]see below[/red]"
