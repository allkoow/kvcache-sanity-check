import csv
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich import box
from kvcache_sanity.models import TestResult

console = Console()


def print_result(result: TestResult, verbose: bool = False) -> None:
    if result.error:
        console.print(f"  iter {result.iteration}: [bold red]ERROR[/] — {result.error}")
        return

    status = "[bold green]PASS[/]" if result.evaluation.passed else "[bold red]FAIL[/]"
    score_pct = f"{result.evaluation.score * 100:.0f}%"
    console.print(f"  iter {result.iteration}: {status}  score={score_pct}  {result.evaluation.reasoning}")

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


def write_accuracy_csv(results: list[TestResult], path: Path) -> None:
    """Write results as a CloudAI-compatible accuracy CSV.

    Columns are ``Task,Correct,Total,Accuracy`` with one row per scenario (Correct =
    passed iterations, Accuracy = Correct/Total as a 0.0-1.0 fraction) plus a final
    OVERALL row aggregating every iteration. CloudAI's AIDynamo report strategy reads
    the Accuracy of the OVERALL row, so the file is meaningful even when the process
    exit code is decoupled from pass/fail (e.g. wrapped with ``|| true``).
    """
    # scenario_id -> [correct, total], insertion-ordered so the CSV mirrors run order.
    per_scenario: dict[str, list[int]] = {}

    for result in results:
        bucket = per_scenario.setdefault(result.scenario_id, [0, 0])
        bucket[1] += 1
        if result.evaluation.passed:
            bucket[0] += 1

    total_correct = sum(correct for correct, _ in per_scenario.values())
    total_count = sum(count for _, count in per_scenario.values())

    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Task", "Correct", "Total", "Accuracy"])

        for scenario_id, (correct, count) in per_scenario.items():
            accuracy = correct / count if count else 0.0
            writer.writerow([scenario_id, correct, count, f"{accuracy:.4f}"])

        overall = total_correct / total_count if total_count else 0.0
        writer.writerow(["OVERALL", total_correct, total_count, f"{overall:.4f}"])
