from __future__ import annotations

import httpx
import openai

from kvcache_sanity.cases import messages_sha256
from kvcache_sanity.models import Case, ResultStatus
from kvcache_sanity.phases import PhaseContext, run_reference_phase
from kvcache_sanity.runner import inject_prefix
from tests.conftest import FakeClient, make_request, make_response

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]


def _case(case_id: str = "early", temperature: float | None = None) -> Case:
    return Case(case_id=case_id, scenario_id=case_id, question="q", messages=MESSAGES,
                messages_sha256=messages_sha256(MESSAGES), temperature=temperature)


def _context(client: FakeClient, reference_retries: int = 2) -> PhaseContext:
    return PhaseContext(target_client=client, model="model", max_tokens=64, temperature=0.1,
                        reference_retries=reference_retries)


def _prefix_of(call: dict) -> str:
    return call["messages"][0]["content"].split("]", 1)[0].lstrip("[")


def _raise_server_error(kwargs: dict):
    raise openai.InternalServerError("boom", response=httpx.Response(500, request=make_request()), body=None)


def test_complete_reference_is_pass():
    client = FakeClient(lambda kwargs: make_response("answer"))

    reference_set = run_reference_phase(_context(client), [_case()])

    (record,) = reference_set.records
    assert record.status == ResultStatus.PASS
    assert record.answer == "answer"
    assert record.prefix == _prefix_of(client.calls[0])
    assert record.attempts[0].complete and record.attempts[0].started_at
    assert (reference_set.model, reference_set.temperature, reference_set.max_tokens) == ("model", 0.1, 64)


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


def test_timed_out_reference_is_timeout():
    def responder(kwargs):
        raise openai.APITimeoutError(request=make_request())

    (record,) = run_reference_phase(_context(FakeClient(responder), reference_retries=0), [_case()]).records

    assert record.status == ResultStatus.TIMEOUT


def test_incomplete_reference_after_all_attempts():
    client = FakeClient(lambda kwargs: make_response("partial", finish_reason="length"))

    (record,) = run_reference_phase(_context(client, reference_retries=1), [_case()]).records

    assert record.status == ResultStatus.REFERENCE_INCOMPLETE
    assert record.error is None


def test_requests_carry_the_case_messages_under_the_attempt_prefix():
    responses = iter([make_response("partial", finish_reason="length"), make_response("answer")])
    client = FakeClient(lambda kwargs: next(responses))
    case = _case()

    run_reference_phase(_context(client), [case])

    for call in client.calls:
        assert call["messages"] == inject_prefix(case.messages, _prefix_of(call))


def test_case_temperature_overrides_the_run_temperature():
    client = FakeClient(lambda kwargs: make_response("answer"))

    run_reference_phase(_context(client), [_case("early"), _case("late", temperature=0.7)])

    assert [call["temperature"] for call in client.calls] == [0.1, 0.7]
