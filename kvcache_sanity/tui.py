"""Interactive TUI for browsing kvcache-check log files.

Usage:
    kvcache-logs runs.jsonl
"""
from pathlib import Path

import click
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, VerticalScroll
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Static,
)

from kvcache_sanity.logger import load_run_logs
from kvcache_sanity.models import RunLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdict(log: RunLog) -> str:
    if log.error:
        return "ERR"
    return "PASS" if log.evaluation.passed else "FAIL"


def _verdict_style(log: RunLog) -> str:
    if log.error:
        return "bold yellow"
    return "bold green" if log.evaluation.passed else "bold red"


def _fmt_messages(messages) -> str:
    lines = []
    for msg in messages:
        role = msg.role.upper()
        content = msg.content.replace("\n", " ")
        lines.append(f"[bold]{role}[/bold]: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detail pane — rebuilt whenever a different run is selected
# ---------------------------------------------------------------------------

class DetailPane(VerticalScroll):
    DEFAULT_CSS = """
    DetailPane {
        width: 3fr;
        border-left: solid $primary-darken-2;
        padding: 1 2;
    }
    """

    def show(self, log: RunLog) -> None:
        self.remove_children()

        verdict = _verdict(log)
        style = _verdict_style(log)
        score_pct = f"{log.evaluation.score * 100:.0f}%"

        self.mount(
            Static(
                f"[bold]{log.scenario_id}[/bold]  iter {log.iteration}  "
                f"[{style}]{verdict}[/]  score={score_pct}",
            )
        )
        meta_parts = [f"[dim]{log.timestamp}[/dim]"]
        if log.target_request_time:
            meta_parts.append(f"[dim]target @ {log.target_request_time}[/dim]")
        if log.target_request_id:
            meta_parts.append(f"[dim]request_id: {log.target_request_id}[/dim]")
        for part in meta_parts:
            self.mount(Static(part))
        self.mount(Static(""))

        if log.error:
            self.mount(Static(f"[bold red]Error:[/] {log.error}"))
            return

        self.mount(Static(f"[bold]Question:[/] {log.question}"))
        self.mount(Static(""))

        self.mount(Collapsible(
            Static(_fmt_messages(log.target_messages), markup=True),
            title="Conversation (target)",
            collapsed=True,
        ))

        target_title = "Target answer"
        if log.target_request_id:
            target_title += f"  (request_id: {log.target_request_id})"
        self.mount(Collapsible(
            Static(log.target_answer),
            title=target_title,
            collapsed=False,
        ))

        self.mount(Collapsible(
            Static(log.reference_answer),
            title="Reference answer  (cache-busted, UUID prefix: "
                  f"{log.reference_prefix[:8]}…)",
            collapsed=False,
        ))

        self.mount(Collapsible(
            Static(_fmt_messages(log.judge_messages), markup=True),
            title="Judge messages",
            collapsed=True,
        ))

        self.mount(Collapsible(
            Static(log.judge_raw_response),
            title="Judge raw response",
            collapsed=False,
        ))

        self.mount(Static(""))
        self.mount(Static(
            f"[bold]Score:[/] {log.evaluation.score:.3f}  "
            f"[bold]Verdict:[/] [{_verdict_style(log)}]{verdict}[/]\n"
            f"[bold]Reasoning:[/] {log.evaluation.reasoning}"
        ))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class LogReviewApp(App):
    TITLE = "kvcache-logs"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]
    CSS = """
    ListView {
        width: 1fr;
        border-right: solid $primary-darken-2;
    }
    ListItem {
        padding: 0 1;
    }
    ListItem.--highlight {
        background: $accent-darken-2;
    }
    """

    def __init__(self, log_path: Path) -> None:
        super().__init__()
        self._log_path = log_path
        self._logs: list[RunLog] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(id="run-list")
            yield DetailPane(id="detail-pane")
        yield Footer()

    def on_mount(self) -> None:
        self._logs = load_run_logs(self._log_path)
        list_view = self.query_one("#run-list", ListView)
        for log in self._logs:
            verdict = _verdict(log)
            style = _verdict_style(log)
            label = (
                f"{log.scenario_id}\n"
                f"  iter {log.iteration}  [{style}]{verdict}[/]  "
                f"{log.evaluation.score * 100:.0f}%"
            )
            list_view.append(ListItem(Label(label, markup=True)))

        self.sub_title = str(self._log_path)

        if self._logs:
            self.query_one("#detail-pane", DetailPane).show(self._logs[0])

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is None:
            return
        idx = self.query_one("#run-list", ListView).index
        if idx is not None and 0 <= idx < len(self._logs):
            self.query_one("#detail-pane", DetailPane).show(self._logs[idx])

    def action_cursor_down(self) -> None:
        self.query_one("#run-list", ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#run-list", ListView).action_cursor_up()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.argument("log_file", type=click.Path(exists=True))
def tui(log_file: str) -> None:
    """Browse a kvcache-check .jsonl log file in an interactive TUI."""
    LogReviewApp(Path(log_file)).run()
