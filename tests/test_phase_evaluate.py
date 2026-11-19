from __future__ import annotations

import csv

from kvcache_sanity.cases import messages_sha256
from kvcache_sanity.evaluator import JudgeConfig
from kvcache_sanity.models import (
    Case,
    ReferenceRecord,
    ReferenceSet,
    ReplayAttempt,
    ReplayRun,
    ResultStatus,
)
from kvcache_sanity.phases import EvaluatePhase, RunState
from kvcache_sanity.report import write_accuracy_csv
from tests.conftest import FakeClient, make_response

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]


def _state(*replays: ReplayRun) -> RunState:
    case = Case(
        case_id="early",
        scenario_id="early",
        question="q",
        messages=MESSAGES,
        messages_sha256=messages_sha256(MESSAGES),
    )
    record = ReferenceRecord(case=case, prefix="ref", answer="reference", status=ResultStatus.PASS)
    reference_set = ReferenceSet(
        created_at="now",
        model="model",
        temperature=0.1,
        max_tokens=64,
        records=[record],
    )

    return RunState(
        cases=[case],
        reference_set=reference_set,
        replays={replay.name: replay for replay in replays},
    )


def _replay(name: str, status: ResultStatus = ResultStatus.PASS) -> ReplayRun:
    attempt = ReplayAttempt(case_id="early", seq=1, prefix="ref", status=status, answer="target")

    return ReplayRun(
        name=name,
        started_at="now",
        finished_at="now",
        attempts=[attempt],
    )


def _judge(client: FakeClient) -> JudgeConfig:
    return JudgeConfig(client=client, model="judge", retries=0)


def _passing_judge(kwargs: dict):
    return make_response('{"reasoning": "ok", "score": 9}')


def test_pass_attempt_is_judged():
    client = FakeClient(_passing_judge)
    state = _state(_replay("warm"))

    EvaluatePhase(_judge(client)).run(state)

    (result,) = state.results

    assert result.status == ResultStatus.PASS
    assert result.phase == "warm"
    assert result.target_answer == "target"
    assert result.reference_answer == "reference"
    assert len(client.calls) == 1


def test_failed_attempt_keeps_its_status_and_is_not_judged():
    client = FakeClient(_passing_judge)
    state = _state(_replay("warm", ResultStatus.TIMEOUT))

    EvaluatePhase(_judge(client)).run(state)

    (result,) = state.results

    assert result.status == ResultStatus.TIMEOUT
    assert not result.evaluation.passed
    assert client.calls == []


def test_csv_has_rows_per_replay(tmp_path):
    state = _state(_replay("first"), _replay("second", ResultStatus.ERROR))
    path = tmp_path / "accuracy_results.csv"

    EvaluatePhase(_judge(FakeClient(_passing_judge))).run(state)
    write_accuracy_csv(state.results, path)

    rows = {row["Task"]: row for row in csv.DictReader(path.open())}

    assert list(rows) == ["first:early", "first:OVERALL", "second:early", "second:OVERALL", "OVERALL"]
    assert rows["first:OVERALL"]["Correct"] == "1"
    assert rows["second:OVERALL"]["Error"] == "1"
    assert rows["OVERALL"]["Total"] == "2"
