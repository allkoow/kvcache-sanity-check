from __future__ import annotations

import csv

from kvcache_sanity.models import EvaluationResult, ResultStatus
from kvcache_sanity.models import TestResult as Result
from kvcache_sanity.report import run_failed, write_accuracy_csv


def _result(status: ResultStatus, passed: bool, scenario_id: str = "early") -> Result:
    return Result(
        scenario_id=scenario_id, iteration=1, question="q", target_answer="", reference_answer="",
        evaluation=EvaluationResult(score=1.0 if passed else 0.0, reasoning="", passed=passed), status=status,
    )


def test_no_results_is_a_failure():
    assert run_failed([])


def test_all_passed_is_not_a_failure():
    assert not run_failed([_result(ResultStatus.PASS, True)])


def test_one_failed_is_a_failure():
    assert run_failed([_result(ResultStatus.PASS, True), _result(ResultStatus.TIMEOUT, False)])


def test_csv_counts_errors_and_timeouts(tmp_path):
    path = tmp_path / "accuracy_results.csv"
    write_accuracy_csv([_result(ResultStatus.ERROR, False), _result(ResultStatus.TIMEOUT, False)], path)

    rows = list(csv.DictReader(path.open()))
    overall = rows[-1]

    assert overall["Task"] == "OVERALL"
    assert overall["Total"] == "2" and overall["Correct"] == "0"
    assert overall["Error"] == "1" and overall["Timeout"] == "1"
    assert overall["Invalid"] == "0"


def test_csv_keeps_existing_column_order(tmp_path):
    path = tmp_path / "accuracy_results.csv"
    write_accuracy_csv([_result(ResultStatus.PASS, True)], path)

    header = next(csv.reader(path.open()))

    assert header == [
        "Task", "Correct", "Total", "Accuracy", "Invalid", "ReferenceIncomplete", "TargetIncomplete", "JudgeError",
        "TargetLoop", "Error", "Timeout",
    ]


def test_csv_without_results_has_an_empty_overall(tmp_path):
    path = tmp_path / "accuracy_results.csv"
    write_accuracy_csv([], path)

    rows = list(csv.DictReader(path.open()))

    assert [row["Task"] for row in rows] == ["OVERALL"]
    assert rows[0]["Total"] == "0" and rows[0]["Accuracy"] == "0.0000"
