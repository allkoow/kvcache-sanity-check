from __future__ import annotations

import threading
import time

import pytest

from kvcache_sanity import engine
from kvcache_sanity.engine import Checkpoint, parse_plan, run_phases
from kvcache_sanity.phases import Phase, PhaseContext, RunState
from kvcache_sanity.state import StateDir
from tests.conftest import FakeClient


class _Recorder(Phase):
    def __init__(self, name: str, ran: list[str], runs_after_interrupt: bool = False) -> None:
        self.name = name
        self.ran = ran
        self.runs_after_interrupt = runs_after_interrupt

    def run(self, state: RunState) -> None:
        self.ran.append(self.name)


def _answer_checkpoint(state_dir: StateDir, name: str, answer: str) -> threading.Thread:
    def answer_when_waiting():
        waiting = state_dir.control_dir / f"{name}.waiting"

        while not waiting.exists():
            time.sleep(0.01)

        (state_dir.control_dir / f"{name}.{answer}").touch()

    thread = threading.Thread(target=answer_when_waiting)
    thread.start()

    return thread


def _steps(ran: list[str]) -> list:
    return [
        _Recorder("before", ran),
        Checkpoint("wait-1"),
        _Recorder("after", ran),
        _Recorder("evaluate", ran, runs_after_interrupt=True),
    ]


def test_plan_names_replays_and_checkpoints_in_order():
    context = PhaseContext(
        target_client=FakeClient(lambda kwargs: None),
        model="model",
        max_tokens=64,
        temperature=0.1,
        reference_retries=0,
    )

    steps = parse_plan("replay, wait ,replay", context, iterations=2)

    assert [step.name for step in steps] == ["replay-1", "wait-1", "replay-2"]

    with pytest.raises(ValueError):
        parse_plan("replay,sleep", context, iterations=2)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("continue", ["before", "after", "evaluate"]),
        ("abort", ["before", "evaluate"]),
    ],
)
def test_checkpoint_is_released_or_aborted_by_its_control_file(tmp_path, monkeypatch, answer, expected):
    monkeypatch.setattr(engine, "POLL_INTERVAL", 0.01)
    state_dir = StateDir(tmp_path)
    ran: list[str] = []

    thread = _answer_checkpoint(state_dir, "wait-1", answer)
    interrupted = run_phases(_steps(ran), RunState(cases=[]), state_dir, checkpoint_timeout=10)
    thread.join()

    assert ran == expected
    assert interrupted == (answer == "abort")
    assert not (state_dir.control_dir / "wait-1.waiting").exists()


def test_checkpoint_timeout_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "POLL_INTERVAL", 0.01)
    ran: list[str] = []

    interrupted = run_phases(_steps(ran), RunState(cases=[]), StateDir(tmp_path), checkpoint_timeout=0.05)

    assert interrupted
    assert ran == ["before", "evaluate"]
