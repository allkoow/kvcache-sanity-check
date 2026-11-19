from __future__ import annotations

import openai

from tests.conftest import make_request, make_response, run_cli


def test_target_timeout_is_a_failed_iteration(tmp_path, monkeypatch):
    calls = {"count": 0}

    def responder(kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return make_response("Guido van Rossum.")
        raise openai.APITimeoutError(request=make_request())

    run = run_cli(tmp_path, monkeypatch, responder, "--iterations", "1")

    assert run.exit_code == 1
    assert run.overall["Total"] == "1" and run.overall["Correct"] == "0"
    assert run.overall["Timeout"] == "1" and run.overall["Error"] == "0"
    assert run.logs[0]["status"] == "TIMEOUT"
    assert run.logs[0]["error"].startswith("APITimeoutError")
    assert calls["count"] == 2  # reference + target; no judge call for a failed target


def test_reference_failures_are_reported_not_hidden(tmp_path, monkeypatch):
    def responder(kwargs):
        raise openai.APIConnectionError(request=make_request())

    run = run_cli(tmp_path, monkeypatch, responder, "--iterations", "1")

    assert run.exit_code == 1
    assert run.overall["Total"] == "1" and run.overall["ReferenceIncomplete"] == "1"
    assert run.logs[0]["status"] == "REFERENCE_INCOMPLETE"
    assert len(run.logs[0]["reference_attempts"]) == 3
