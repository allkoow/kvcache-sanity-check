from __future__ import annotations

import csv
import json

import openai
from click.testing import CliRunner

from kvcache_sanity import main
from tests.conftest import FakeClient, make_request, make_response

SCENARIO_YAML = """\
scenarios:
  - id: recall
    description: "one document, one question"
    document_ids: [doc_s1_python]
    question: "Who created Python?"
"""


def _run_cli(tmp_path, monkeypatch, responder) -> tuple[int, dict, list[dict]]:
    scenarios_file = tmp_path / "scenarios.yaml"
    scenarios_file.write_text(SCENARIO_YAML)
    accuracy_csv = tmp_path / "accuracy_results.csv"
    log_file = tmp_path / "runs.jsonl"
    client = FakeClient(responder)
    monkeypatch.setattr(main, "make_client", lambda *args, **kwargs: client)

    outcome = CliRunner().invoke(main.cli, [
        "--target-url", "http://server", "--model", "model", "--scenarios-file", str(scenarios_file),
        "--iterations", "1", "--accuracy-csv", str(accuracy_csv), "--log-file", str(log_file),
    ])

    overall = list(csv.DictReader(accuracy_csv.open()))[-1]
    logs = [json.loads(line) for line in log_file.read_text().splitlines()]
    return outcome.exit_code, overall, logs


def test_target_timeout_is_a_failed_iteration(tmp_path, monkeypatch):
    calls = {"count": 0}

    def responder(kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return make_response("Guido van Rossum.")
        raise openai.APITimeoutError(request=make_request())

    exit_code, overall, logs = _run_cli(tmp_path, monkeypatch, responder)

    assert exit_code == 1
    assert overall["Total"] == "1" and overall["Correct"] == "0"
    assert logs[0]["status"] == "TIMEOUT"
    assert logs[0]["error"].startswith("APITimeoutError")
    assert calls["count"] == 2  # reference + target; no judge call for a failed target


def test_reference_failures_are_reported_not_hidden(tmp_path, monkeypatch):
    def responder(kwargs):
        raise openai.APIConnectionError(request=make_request())

    exit_code, overall, logs = _run_cli(tmp_path, monkeypatch, responder)

    assert exit_code == 1
    assert overall["Total"] == "1" and overall["ReferenceIncomplete"] == "1"
    assert logs[0]["status"] == "REFERENCE_INCOMPLETE"
    assert len(logs[0]["reference_attempts"]) == 3
