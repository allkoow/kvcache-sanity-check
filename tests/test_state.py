from __future__ import annotations

import pytest

from kvcache_sanity.models import Case, ReferenceAttemptLog, ReferenceRecord, ReferenceSet, ResultStatus
from kvcache_sanity.state import StateDir

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]


def _reference_set() -> ReferenceSet:
    case = Case(case_id="early", scenario_id="early", question="q", messages=MESSAGES, messages_sha256="0" * 64)
    attempt = ReferenceAttemptLog(
        prefix="abc", answer="answer", finish_reason="stop", completion_tokens=3, complete=True,
        started_at="2026-11-19T10:00:00+00:00", finished_at="2026-11-19T10:00:05+00:00",
    )
    record = ReferenceRecord(case=case, prefix="abc", answer="answer", status=ResultStatus.PASS, attempts=[attempt])
    return ReferenceSet(created_at="2026-11-19T10:00:05+00:00", model="model", temperature=0.1, max_tokens=64,
                        records=[record])


def test_reference_set_round_trip(tmp_path):
    state = StateDir(tmp_path / "state")
    reference_set = _reference_set()

    state.save_reference_set(reference_set)

    assert state.load_reference_set().model_dump() == reference_set.model_dump()
    assert [path.name for path in (tmp_path / "state").iterdir()] == ["reference_set.json"]


def test_missing_reference_set_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        StateDir(tmp_path).load_reference_set()


def test_state_subdirectories(tmp_path):
    state = StateDir(tmp_path)

    assert state.control_dir == tmp_path / "control"
    assert state.metrics_dir == tmp_path / "metrics"
