from __future__ import annotations

import httpx
import openai

from kvcache_sanity.cases import messages_sha256
from kvcache_sanity.models import Case, ReferenceRecord, ReferenceSet, ReplayRun, ResultStatus
from kvcache_sanity.phases import PhaseContext, ReplayPhase, RunState
from kvcache_sanity.runner import inject_prefix
from tests.conftest import FakeClient, make_request, make_response

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]


def _record(case_id: str, status: ResultStatus = ResultStatus.PASS):
    case = Case(
        case_id=case_id,
        scenario_id=case_id,
        question="q",
        messages=MESSAGES,
        messages_sha256=messages_sha256(MESSAGES),
    )

    return ReferenceRecord(case=case, prefix=f"ref-{case_id}", answer="answer", status=status)


def _reference_set(*records: ReferenceRecord) -> ReferenceSet:
    return ReferenceSet(
        created_at="now",
        model="model",
        temperature=0.1,
        max_tokens=64,
        records=list(records),
    )


def _context(client: FakeClient) -> PhaseContext:
    return PhaseContext(
        target_client=client,
        model="model",
        max_tokens=64,
        temperature=0.1,
        reference_retries=2,
    )


def _replay(
    client: FakeClient,
    reference_set: ReferenceSet,
    name: str,
    **options,
) -> ReplayRun:
    state = RunState(cases=[], reference_set=reference_set)

    ReplayPhase(_context(client), name, **options).run(state)

    return state.replays[name]


def _answer(kwargs: dict):
    return make_response("answer")


def _raise_server_error():
    response = httpx.Response(500, request=make_request())

    raise openai.InternalServerError("boom", response=response, body=None)


def test_reuse_sends_every_iteration_under_the_reference_prefix():
    client = FakeClient(_answer)
    reference_set = _reference_set(_record("early"))

    replay = _replay(client, reference_set, "warm", iterations=3)

    assert len(client.calls) == 3
    assert [attempt.seq for attempt in replay.attempts] == [1, 2, 3]
    assert {attempt.prefix for attempt in replay.attempts} == {"ref-early"}
    assert all(attempt.status == ResultStatus.PASS for attempt in replay.attempts)
    assert all(call["messages"] == inject_prefix(MESSAGES, "ref-early") for call in client.calls)
    assert replay.name == "warm"
    assert replay.started_at <= replay.finished_at


def test_failed_iteration_does_not_stop_the_others():
    calls = {"count": 0}

    def responder(kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            _raise_server_error()

        return make_response("answer")

    reference_set = _reference_set(_record("early"))

    replay = _replay(FakeClient(responder), reference_set, "warm", iterations=3)

    statuses = [attempt.status for attempt in replay.attempts]
    failed = replay.attempts[1]

    assert statuses == [ResultStatus.PASS, ResultStatus.ERROR, ResultStatus.PASS]
    assert failed.error.startswith("InternalServerError")
    assert failed.started_at and failed.finished_at


def test_case_without_a_complete_reference_is_carried_not_sent():
    client = FakeClient(_answer)
    reference_set = _reference_set(_record("early", ResultStatus.TIMEOUT), _record("late"))

    replay = _replay(client, reference_set, "warm", iterations=3)

    early = [(attempt.seq, attempt.status) for attempt in replay.attempts if attempt.case_id == "early"]

    assert early == [(1, ResultStatus.TIMEOUT)]
    assert len(client.calls) == 3


def test_attempts_follow_reference_order_then_seq():
    reference_set = _reference_set(_record("late"), _record("early"))

    replay = _replay(FakeClient(_answer), reference_set, "warm", iterations=2)

    order = [(attempt.case_id, attempt.seq) for attempt in replay.attempts]

    assert order == [("late", 1), ("late", 2), ("early", 1), ("early", 2)]

