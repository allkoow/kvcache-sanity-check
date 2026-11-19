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
    assert run.overall["Total"] == "1"
    assert run.overall["Correct"] == "0"
    assert run.overall["Timeout"] == "1"
    assert run.overall["Error"] == "0"
    # Reference and target only: a failed target is never judged.
    assert calls["count"] == 2
