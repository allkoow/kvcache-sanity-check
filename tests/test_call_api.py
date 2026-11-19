from __future__ import annotations

import httpx
import openai

from kvcache_sanity.evaluator import JudgeConfig, evaluate_answers
from kvcache_sanity.runner import _call_api, make_client
from tests.conftest import make_request

MESSAGES = [{"role": "user", "content": "question"}]


def _raise_server_error(kwargs: dict):
    response = httpx.Response(500, request=make_request())

    raise openai.InternalServerError("boom", response=response, body=None)


def _raise_timeout(kwargs: dict):
    raise openai.APITimeoutError(request=make_request())


def test_status_error_is_returned_not_raised(fake_client):
    client = fake_client(_raise_server_error)

    run = _call_api(client, "model", MESSAGES, 16, empty_retries=0)

    assert run.answer == ""
    assert run.error.startswith("InternalServerError")
    assert not run.timed_out
    assert run.started_at and run.finished_at
    assert len(client.calls) == 1


def test_timeout_is_flagged(fake_client):
    run = _call_api(fake_client(_raise_timeout), "model", MESSAGES, 16, empty_retries=0)

    assert run.timed_out
    assert run.error.startswith("APITimeoutError")


def test_client_never_retries():
    client = make_client("http://server/", "EMPTY", timeout=30.0)

    assert client.max_retries == 0
    assert client.timeout == 30.0
    assert str(client.base_url) == "http://server/v1/"


def test_judge_request_failure_is_a_retried_judge_error(fake_client):
    client = fake_client(_raise_server_error)
    judge = JudgeConfig(client=client, model="model", retries=1)

    trace = evaluate_answers("question", "target", "reference", judge)

    assert len(client.calls) == 2
    assert trace.judge_error.startswith("request failed: InternalServerError")
    assert not trace.result.passed
    assert [attempt.problem is not None for attempt in trace.attempt_logs] == [True, True]
