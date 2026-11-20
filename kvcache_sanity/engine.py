from __future__ import annotations

import json
import time
from dataclasses import dataclass

from rich.console import Console
from rich.markup import escape

from kvcache_sanity.phases import Phase, PhaseContext, ReplayPhase, RunState
from kvcache_sanity.runner import utc_now
from kvcache_sanity.state import StateDir, write_atomic

console = Console()

DEFAULT_CHECKPOINT_TIMEOUT = 300.0
POLL_INTERVAL = 1.0


@dataclass(frozen=True)
class Checkpoint:
    """A pause between phases until the orchestrating test releases or aborts the run.

    The engine writes control/<name>.waiting; the test answers with control/<name>.continue
    to go on or control/<name>.abort to stop.
    """

    name: str


Step = Phase | Checkpoint


def parse_plan(plan: str, context: PhaseContext, iterations: int) -> list[Step]:
    """Steps between the reference and evaluate phases, e.g. "replay,wait,replay".

    Replays are named replay-1, replay-2, ... and checkpoints wait-1, wait-2, ... in plan order.
    """

    steps: list[Step] = []
    replays = 0
    waits = 0

    for item in plan.split(","):
        kind = item.strip()

        if kind == "replay":
            replays += 1
            steps.append(ReplayPhase(context, f"replay-{replays}", iterations))
        elif kind == "wait":
            waits += 1
            steps.append(Checkpoint(f"wait-{waits}"))
        else:
            raise ValueError(f"unknown plan step {kind!r}: use replay or wait")

    if not replays:
        raise ValueError("the plan needs at least one replay")

    return steps


def save_state(state: RunState, state_dir: StateDir) -> None:
    if state.reference_set is not None:
        state_dir.save_reference_set(state.reference_set)

    for replay in state.replays.values():
        state_dir.save_replay(replay)


def wait_for_checkpoint(checkpoint: Checkpoint, state_dir: StateDir, timeout: float) -> bool:
    """Wait until the test releases the checkpoint; False when it aborts or the timeout expires."""

    control_dir = state_dir.control_dir
    waiting = control_dir / f"{checkpoint.name}.waiting"
    release = control_dir / f"{checkpoint.name}.continue"
    abort = control_dir / f"{checkpoint.name}.abort"

    release.unlink(missing_ok=True)
    abort.unlink(missing_ok=True)

    marker = {
        "name": checkpoint.name,
        "since": utc_now(),
        "timeout": timeout,
    }
    write_atomic(waiting, json.dumps(marker))

    console.print(f"  waiting for {escape(str(release))} (or .abort), timeout {timeout:.0f} s")

    deadline = time.monotonic() + timeout

    try:
        while time.monotonic() < deadline:
            if release.exists():
                console.print("  released")
                return True

            if abort.exists():
                console.print("  [bold yellow]aborted[/]")
                return False

            time.sleep(POLL_INTERVAL)
    finally:
        waiting.unlink(missing_ok=True)

    console.print("  [bold yellow]timed out[/]")

    return False


def run_phases(
    steps: list[Step],
    state: RunState,
    state_dir: StateDir | None = None,
    checkpoint_timeout: float = DEFAULT_CHECKPOINT_TIMEOUT,
) -> bool:
    """Run the steps in order and return whether the run was interrupted.

    Ctrl+C, SIGTERM, an aborted checkpoint or one that timed out stops the run; of the remaining
    phases only those with runs_after_interrupt still run, on what the earlier phases finished.
    With state_dir the artifacts are saved after every phase, interrupted ones included.
    Checkpoints need state_dir: their control files live in it.
    """

    interrupted = False

    for step in steps:
        if interrupted and not (isinstance(step, Phase) and step.runs_after_interrupt):
            continue

        console.print(f"[bold cyan]{escape(step.name)}[/]")

        try:
            if isinstance(step, Checkpoint):
                if state_dir is None:
                    raise ValueError(f"checkpoint {step.name} needs a state dir")

                interrupted = not wait_for_checkpoint(step, state_dir, checkpoint_timeout)
            else:
                step.run(state)
        except KeyboardInterrupt:
            interrupted = True
            console.print(f"[bold yellow]Interrupted[/] during {escape(step.name)}")
        finally:
            if state_dir and isinstance(step, Phase):
                save_state(state, state_dir)

        console.print()

    return interrupted
