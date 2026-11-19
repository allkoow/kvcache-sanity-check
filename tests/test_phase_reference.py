from __future__ import annotations

import httpx
import openai

from kvcache_sanity.cases import messages_sha256
from kvcache_sanity.models import Case, ResultStatus
from kvcache_sanity.phases import PhaseContext, run_reference_phase
from tests.conftest import FakeClient, make_request, make_response

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]


def _case() -> Case:
    return Case(
        case_id="early",
        scenario_id="early",
        question="q",
        messages=MESSAGES,
        messages_sha256=messages_sha256(MESSAGES),
    )


def _context(client: FakeClient) -> PhaseContext:
    return PhaseContext(
        target_client=client,
        model="model",
        max_tokens=64,
        temperature=0.1,
        reference_retries=2,
    )


def _prefix_of(call: dict) -> str:
    return call["messages"][0]["content"].split("]", 1)[0].lstrip("[")


def _raise_server_error(kwargs: dict):
    response = httpx.Response(500, request=make_request())

    raise openai.InternalServerError("boom", response=response, body=None)


def test_complete_reference_is_pass():
    client = FakeClient(lambda kwargs: make_response("answer"))

    reference_set = run_reference_phase(_context(client), [_case()])

    (record,) = reference_set.records

    assert record.status == ResultStatus.PASS
    assert record.answer == "answer"
    assert record.prefix == _prefix_of(client.calls[0])
    assert record.attempts[0].complete
    assert record.attempts[0].started_at
    assert reference_set.model == "model"
    assert reference_set.temperature == 0.1
    assert reference_set.max_tokens == 64


def test_truncated_reference_is_retried_under_a_fresh_prefix():
    responses = iter([make_response("partial", finish_reason="length"), make_response("answer")])
    client = FakeClient(lambda kwargs: next(responses))

    (record,) = run_reference_phase(_context(client), [_case()]).records

    first, second = (_prefix_of(call) for call in client.calls)

    assert record.status == ResultStatus.PASS
    assert first != second
    assert record.prefix == second
    assert [attempt.prefix for attempt in record.attempts] == [first, second]
    assert [attempt.complete for attempt in record.attempts] == [False, True]


def test_failed_reference_is_kept_with_its_error():
    client = FakeClient(_raise_server_error)

    (record,) = run_reference_phase(_context(client), [_case()]).records

    assert record.status == ResultStatus.ERROR
    assert record.answer == ""
    assert record.error.startswith("InternalServerError")
    assert len(record.attempts) == 3
    assert all(attempt.error for attempt in record.attempts)
