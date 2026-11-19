import csv
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.markup import escape
from kvcache_sanity.models import ResultStatus, TestResult

console = Console()


def print_result(result: TestResult, verbose: bool = False) -> None:
    if result.error:
        console.print(f"  iter {result.iteration}: [bold red]ERROR[/] — {result.error}")
        return

    if result.status == ResultStatus.REFERENCE_INCOMPLETE:
        console.print(
            f"  iter {result.iteration}: [bold yellow]REFERENCE_INCOMPLETE[/] — {escape(result.evaluation.reasoning)}"
        )
        return

    if result.status == ResultStatus.TARGET_INCOMPLETE:
        status = "[bold red]TARGET_INCOMPLETE[/]"
    elif result.status == ResultStatus.JUDGE_ERROR:
        status = "[bold yellow]JUDGE_ERROR[/]"
    else:
        status = "[bold green]PASS[/]" if result.evaluation.passed else "[bold red]FAIL[/]"
    score_pct = f"{result.evaluation.score * 100:.0f}%"
    console.print(f"  iter {result.iteration}: {status}  score={score_pct}  {escape(result.evaluation.reasoning)}")

    if verbose or not result.evaluation.passed:
        if result.target_request_id:
            console.print(f"    [dim]request_id: {result.target_request_id}[/]")
        console.print(f"    [dim]Target   : {result.target_answer[:300].replace(chr(10), ' ')}[/]")
        console.print(f"    [dim]Reference: {result.reference_answer[:300].replace(chr(10), ' ')}[/]")


def print_summary(results: list[TestResult]) -> None:
    total = len(results)
    if total == 0:
        console.print("[yellow]No results to summarize.[/]")
        return

    passed = sum(1 for r in results if r.evaluation.passed)
    avg_score = sum(r.evaluation.score for r in results) / total

    if passed == total:
        color = "green"
    elif passed > total // 2:
        color = "yellow"
    else:
        color = "red"

    console.print()
    console.print(Panel(
        f"[bold {color}]{passed}/{total} passed[/]   avg score: {avg_score * 100:.1f}%",
        title="[bold]Summary[/]",
        box=box.ROUNDED,
    ))


_CSV_HEADER = [
    "Task", "Correct", "Total", "Accuracy", "Invalid", "ReferenceIncomplete", "TargetIncomplete", "JudgeError",
    "TargetLoop",
]


def _csv_row(task: str, results: list[TestResult]) -> list:
    statuses = Counter(result.status for result in results)
    correct = sum(1 for result in results if result.evaluation.passed)

    total = len(results)
    accuracy = correct / total if total else 0.0

    reference_incomplete = statuses[ResultStatus.REFERENCE_INCOMPLETE]
    judge_error = statuses[ResultStatus.JUDGE_ERROR]

    target_loop = sum(
        1 for result in results
        if result.status == ResultStatus.TARGET_INCOMPLETE and result.target_truncation == "loop"
    )

    return [
        task, correct, total, f"{accuracy:.4f}",
        reference_incomplete + judge_error, reference_incomplete, statuses[ResultStatus.TARGET_INCOMPLETE], judge_error,
        target_loop
    ]


def write_accuracy_csv(results: list[TestResult], path: Path) -> None:
    """Write results as a CloudAI-compatible accuracy CSV.

    Columns are ``Task,Correct,Total,Accuracy,Invalid,ReferenceIncomplete,TargetIncomplete,JudgeError,TargetLoop``
    with one row per scenario (Correct = passed iterations, Accuracy = Correct/Total as a
    0.0-1.0 fraction) plus a final OVERALL row aggregating every iteration. CloudAI's
    AIDynamo report strategy reads the Accuracy of the OVERALL row by column name, so the
    file is meaningful even when the process exit code is decoupled from pass/fail (e.g.
    wrapped with ``|| true``).

    Invalid counts iterations that say nothing about the cache path; ReferenceIncomplete
    and JudgeError are its breakdown. Invalid iterations are still part of Total and lower
    Accuracy, so consumers that ignore the extra columns keep seeing them as failures.

    TargetIncomplete counts truncated or empty target answers. They are incorrect answers,
    not invalid ones: part of Total, never of Correct or Invalid. TargetLoop is the subset of
    TargetIncomplete truncated while repeating itself next to a complete reference, a possible
    sign of corrupted KV blocks rather than a too small token budget.
    """
    # Insertion-ordered so the CSV mirrors run order.
    per_scenario: dict[str, list[TestResult]] = {}

    for result in results:
        per_scenario.setdefault(result.scenario_id, []).append(result)

    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(_CSV_HEADER)

        for scenario_id, scenario_results in per_scenario.items():
            writer.writerow(_csv_row(scenario_id, scenario_results))

        writer.writerow(_csv_row("OVERALL", results))
