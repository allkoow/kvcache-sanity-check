from __future__ import annotations

import os
import signal

from tests.conftest import is_judge_request, make_response, run_cli

JUDGE_PASS = '{"reasoning": "same", "score": 9}'


def _interrupt_on_second_target(interrupt):
    target_calls = {"count": 0}

    def responder(kwargs):
        if is_judge_request(kwargs):
            return make_response(JUDGE_PASS)
        target_calls["count"] += 1
        # Reference first, then the first target iteration; the second target iteration is interrupted.
        if target_calls["count"] == 3:
            interrupt()
        return make_response("Guido van Rossum.")

    return responder


def _keyboard_interrupt():
    raise KeyboardInterrupt


def _sigterm():
    os.kill(os.getpid(), signal.SIGTERM)


def test_ctrl_c_keeps_finished_iterations_in_csv(tmp_path, monkeypatch):
    run = run_cli(tmp_path, monkeypatch, _interrupt_on_second_target(_keyboard_interrupt), "--iterations", "3")

    assert run.exit_code == 1
    assert run.overall["Total"] == "1" and run.overall["Correct"] == "1"
    assert "Interrupted" in run.output


def test_sigterm_keeps_finished_iterations_in_csv(tmp_path, monkeypatch):
    handler_before = signal.getsignal(signal.SIGTERM)

    run = run_cli(tmp_path, monkeypatch, _interrupt_on_second_target(_sigterm), "--iterations", "3")

    assert run.exit_code == 1
    assert run.overall["Total"] == "1" and run.overall["Correct"] == "1"
    assert signal.getsignal(signal.SIGTERM) is handler_before


def test_csv_is_current_after_every_result(tmp_path, monkeypatch):
    csv_path = tmp_path / "accuracy_results.csv"
    seen_totals: list[str] = []

    def responder(kwargs):
        if is_judge_request(kwargs):
            return make_response(JUDGE_PASS)
        if csv_path.exists():
            seen_totals.append(csv_path.read_text().splitlines()[-1].split(",")[2])
        return make_response("Guido van Rossum.")

    run = run_cli(tmp_path, monkeypatch, responder, "--iterations", "3")

    assert run.exit_code == 0
    # Before the 2nd and 3rd target request the CSV already holds the earlier iterations.
    assert seen_totals == ["1", "2"]
