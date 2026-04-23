import json
from datetime import datetime, timezone
from pathlib import Path

from kvcache_sanity.models import (
    EvaluationResult,
    EvaluationTrace,
    MessageLog,
    RunLog,
    RunResult,
    Scenario,
    TestResult,
)

_DOC_TRUNCATE_LIMIT = 200


def _truncate(text: str, limit: int = _DOC_TRUNCATE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[{len(text) - limit} chars truncated]"


def _truncate_messages(messages: list[dict]) -> list[MessageLog]:
    """Copy messages, truncating document content so logs stay readable."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        # Document turns start with "Document N —"; truncate only those.
        if msg.get("role") == "user" and content.startswith("Document "):
            content = _truncate(content)
        result.append(MessageLog(role=msg["role"], content=content))
    return result


class RunLogger:
    def __init__(self, path: Path) -> None:
        self._path = path

    def log(
        self,
        scenario: Scenario,
        iteration: int,
        target: RunResult,
        reference: RunResult,
        trace: EvaluationTrace,
        error: str | None = None,
    ) -> None:
        run_log = RunLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            scenario_id=scenario.id,
            iteration=iteration,
            question=scenario.question,
            target_messages=_truncate_messages(target.messages),
            reference_prefix=reference.unique_prefix or "",
            target_answer=target.answer,
            reference_answer=reference.answer,
            target_request_id=target.request_id,
            reference_request_id=reference.request_id,
            target_request_time=target.request_time,
            judge_messages=_truncate_messages(trace.judge_messages),
            judge_raw_response=trace.judge_raw_response,
            evaluation=trace.result,
            error=error,
        )
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(run_log.model_dump_json() + "\n")


def load_run_logs(path: Path) -> list[RunLog]:
    """Read all RunLog entries from a .jsonl file, skipping malformed lines."""
    logs = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(RunLog.model_validate_json(line))
            except Exception:
                pass  # silently skip bad lines; TUI shows a warning row instead
    return logs
