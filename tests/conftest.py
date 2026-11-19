from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner

SCENARIO_YAML = """\
scenarios:
  - id: recall
    description: "one document, one question"
    document_ids: [doc_s1_python]
    question: "Who created Python?"
"""


def make_response(content: str, finish_reason: str = "stop", request_id: str = "req-1") -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(completion_tokens=len(content.split()), completion_tokens_details=None)

    return SimpleNamespace(id=request_id, choices=[choice], usage=usage)


def make_request() -> httpx.Request:
    return httpx.Request("POST", "http://server/v1/chat/completions")


class FakeCompletions:
    def __init__(self, responder: Callable[[dict], SimpleNamespace]) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)

        return self.responder(kwargs)


class FakeClient:
    """Stands in for openai.OpenAI: `responder` gets the request kwargs and returns or raises."""

    def __init__(self, responder: Callable[[dict], SimpleNamespace]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responder))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


@pytest.fixture
def fake_client() -> type[FakeClient]:
    return FakeClient


def is_judge_request(kwargs: dict) -> bool:
    return "response_format" in kwargs


@dataclass
class CliRun:
    exit_code: int
    output: str
    csv_rows: list[dict]
    client: FakeClient

    @property
    def overall(self) -> dict:
        return self.csv_rows[-1]


def run_cli(tmp_path: Path, monkeypatch, responder: Callable[[dict], SimpleNamespace], *args: str) -> CliRun:
    """Run kvcache-check on a one-scenario file with every request answered by `responder`."""

    from kvcache_sanity import main

    scenarios_file = tmp_path / "scenarios.yaml"
    scenarios_file.write_text(SCENARIO_YAML)
    accuracy_csv = tmp_path / "accuracy_results.csv"

    client = FakeClient(responder)
    monkeypatch.setattr(main, "make_client", lambda *_args, **_kwargs: client)

    cli_args = [
        "--target-url", "http://server",
        "--model", "model",
        "--scenarios-file", str(scenarios_file),
        "--accuracy-csv", str(accuracy_csv),
        *args,
    ]
    outcome = CliRunner().invoke(main.cli, cli_args)

    csv_rows = list(csv.DictReader(accuracy_csv.open())) if accuracy_csv.exists() else []

    return CliRun(outcome.exit_code, outcome.output, csv_rows, client)
