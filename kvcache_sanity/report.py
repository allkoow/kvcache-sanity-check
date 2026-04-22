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
