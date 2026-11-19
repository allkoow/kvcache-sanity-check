from __future__ import annotations

from rich.console import Console
from rich.markup import escape

from kvcache_sanity.phases import Phase, RunState
from kvcache_sanity.state import StateDir

console = Console()


def save_state(state: RunState, state_dir: StateDir) -> None:
    if state.reference_set is not None:
        state_dir.save_reference_set(state.reference_set)

    for replay in state.replays.values():
        state_dir.save_replay(replay)


def run_phases(phases: list[Phase], state: RunState, state_dir: StateDir | None = None) -> bool:
    """Run the phases in order and return whether the run was interrupted.

    Ctrl+C or SIGTERM stops the running phase; of the remaining ones only those with
    runs_after_interrupt still run, on what the earlier phases finished. With state_dir the
    artifacts are saved after every phase, interrupted ones included.
    """

    interrupted = False

    for phase in phases:
        if interrupted and not phase.runs_after_interrupt:
            continue

        console.print(f"[bold cyan]{escape(phase.name)}[/]")

        try:
            phase.run(state)
        except KeyboardInterrupt:
            interrupted = True
            console.print(f"[bold yellow]Interrupted[/] during {escape(phase.name)}")
        finally:
            if state_dir:
                save_state(state, state_dir)

        console.print()

    return interrupted
