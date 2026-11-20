from __future__ import annotations

import csv

from kvcache_sanity.models import EvaluationResult, ResultStatus
from kvcache_sanity.models import TestResult as Result
from kvcache_sanity.report import run_failed, write_accuracy_csv


def _result(status: ResultStatus, passed: bool, scenario_id: str = "early") -> Result:
    evaluation = EvaluationResult(score=1.0 if passed else 0.0, reasoning="", passed=passed)

    return Result(
        scenario_id=scenario_id,
        iteration=1,
        question="q",
        target_answer="",
        reference_answer="",
        evaluation=evaluation,
        status=status,
    )


def test_no_results_is_a_failure():
    assert run_failed([])


def test_csv_counts_failed_requests_and_invalid_references(tmp_path):
    path = tmp_path / "accuracy_results.csv"
    results = [
        _result(ResultStatus.ERROR, False),
        _result(ResultStatus.TIMEOUT, False),
        _result(ResultStatus.REFERENCE_MISSING_FACTS, False),
    ]

    write_accuracy_csv(results, path)

    overall = list(csv.DictReader(path.open()))[-1]

    assert overall["Task"] == "OVERALL"
    assert overall["Total"] == "3"
    assert overall["Correct"] == "0"
    assert overall["Error"] == "1"
    assert overall["Timeout"] == "1"
    # Failed requests are incorrect answers; only the reference without the key facts is invalid.
    assert overall["Invalid"] == "1"
    assert overall["ReferenceMissingFacts"] == "1"
