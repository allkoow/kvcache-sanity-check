from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from openai import OpenAI
from rich.console import Console
from rich.markup import escape

from kvcache_sanity.logger import reference_attempt_log
from kvcache_sanity.models import (
    Case,
    PrefixMode,
    ReferenceRecord,
    ReferenceSet,
    ReplayAttempt,
    ReplayRun,
    ResultStatus,
    RunResult,
)
from kvcache_sanity.runner import (
    _call_api,
    describe_generation,
    inject_prefix,
    is_complete,
    prepare_reference,
    utc_now,
)

console = Console()


@dataclass(frozen=True)
class PhaseContext:
    """Target server settings shared by the phases that send the evaluated conversations."""

    target_client: OpenAI
    model: str
    max_tokens: int
    temperature: float
    reference_retries: int


def _case_temperature(context: PhaseContext, case: Case) -> float:
    if case.temperature is not None:
        return case.temperature

    return context.temperature


def _generation_status(run: RunResult, incomplete: ResultStatus) -> ResultStatus:
    if is_complete(run):
        return ResultStatus.PASS

    if run.error:
        return ResultStatus.TIMEOUT if run.timed_out else ResultStatus.ERROR

    return incomplete


def _send(context: PhaseContext, case: Case, prefix: str, label: str) -> RunResult:
    run = _call_api(
        context.target_client,
        context.model,
        inject_prefix(case.messages, prefix),
        context.max_tokens,
        label,
        temperature=_case_temperature(context, case),
        empty_retries=0,
    )
    run.unique_prefix = prefix

    return run


def _reference_record(context: PhaseContext, case: Case) -> ReferenceRecord:
    label = f"reference, case={case.case_id}"

    def generate(prefix: str) -> RunResult:
        return _send(context, case, prefix, label)

    attempts = prepare_reference(generate, str(uuid.uuid4()), retries=context.reference_retries, label=label)
    accepted = attempts[-1]
    status = _generation_status(accepted, ResultStatus.REFERENCE_INCOMPLETE)

    console.print(
        f"  reference [bold]{escape(case.case_id)}[/]: {status.value} "
        f"({len(attempts)} attempt(s), {escape(describe_generation(accepted))})"
    )

    return ReferenceRecord(
        case=case,
        prefix=accepted.unique_prefix or "",
        answer=accepted.answer,
        status=status,
        attempts=[reference_attempt_log(attempt) for attempt in attempts],
        error=accepted.error,
    )


def _replay_prefixes(
    reference_set: ReferenceSet,
    prefix_mode: PrefixMode,
    source: ReplayRun | None,
) -> dict[str, str]:
    """Cache-busting prefix per case_id for a replay."""

    if prefix_mode == PrefixMode.REUSE:
        return {record.case.case_id: record.prefix for record in reference_set.records}

    if prefix_mode == PrefixMode.FRESH:
        return {record.case.case_id: str(uuid.uuid4()) for record in reference_set.records}

    if source is None:
        raise ValueError("prefix mode 'from' needs the source replay")

    return {attempt.case_id: attempt.prefix for attempt in source.attempts}


def _carried_attempt(record: ReferenceRecord) -> ReplayAttempt:
    return ReplayAttempt(
        case_id=record.case.case_id,
        seq=1,
        prefix=record.prefix,
        status=record.status,
        error=record.error,
    )


def _replay_attempt(context: PhaseContext, name: str, prefix: str, seq: int, record: ReferenceRecord) -> ReplayAttempt:
    case = record.case

    run = _send(context, case, prefix, f"replay={name}, case={case.case_id}, seq={seq}")
    status = _generation_status(run, ResultStatus.TARGET_INCOMPLETE)

    console.print(
        f"  {escape(name)} [bold]{escape(case.case_id)}[/] #{seq}: {status.value} "
        f"({escape(describe_generation(run))})"
    )

    return ReplayAttempt(
        case_id=case.case_id,
        seq=seq,
        prefix=prefix,
        started_at=run.started_at,
        finished_at=run.finished_at,
        status=status,
        answer=run.answer,
        finish_reason=run.finish_reason,
        completion_tokens=run.completion_tokens,
        request_id=run.request_id,
        truncation=run.truncation,
        error=run.error,
    )


def run_reference_phase(context: PhaseContext, cases: list[Case]) -> ReferenceSet:
    """Generate the reference of every case under a fresh cache-busting prefix (a full prefill).

    Every case gets a record, failed ones included, so later phases see the whole case set.
    The accepted prefix is the one later replays reuse to read the blocks this phase stored.
    """

    return ReferenceSet(
        created_at=utc_now(),
        model=context.model,
        temperature=context.temperature,
        max_tokens=context.max_tokens,
        records=[_reference_record(context, case) for case in cases],
    )


def run_replay_phase(
    context: PhaseContext,
    reference_set: ReferenceSet,
    name: str,
    prefix_mode: PrefixMode,
    iterations: int,
    duration: float | None = None,
    source: ReplayRun | None = None,
) -> ReplayRun:
    """Send the conversation of every case with a complete reference, without judging the answers.

    prefix_mode picks the cache-busting prefix per case (see PrefixMode); PrefixMode.FROM takes
    the prefixes of the source replay.

    Requests go one at a time, in rounds of one request per case. With duration, rounds repeat
    until it has elapsed (at least one), capped by iterations when that is non-zero.

    A case without a complete reference is not sent; it gets one attempt carrying the
    reference status, so no case drops out of the results.
    """

    prefixes = _replay_prefixes(reference_set, prefix_mode, source)
    records = reference_set.records

    sendable = [record for record in records if record.status == ResultStatus.PASS]
    attempts = [_carried_attempt(record) for record in records if record.status != ResultStatus.PASS]

    missing = [record.case.case_id for record in sendable if record.case.case_id not in prefixes]
    if missing:
        raise ValueError(f"prefix mode {prefix_mode.value!r}: no prefix for case(s) {', '.join(missing)}")

    started_at = utc_now()
    deadline = time.monotonic() + duration if duration is not None else None

    def more_rounds(done: int) -> bool:
        if iterations and done >= iterations:
            return False

        if deadline is None:
            return done < iterations

        return done == 0 or time.monotonic() < deadline

    rounds = 0

    while sendable and more_rounds(rounds):
        rounds += 1

        for record in sendable:
            attempts.append(_replay_attempt(context, name, prefixes[record.case.case_id], rounds, record))

    order = {record.case.case_id: index for index, record in enumerate(records)}
    attempts.sort(key=lambda attempt: (order[attempt.case_id], attempt.seq))

    return ReplayRun(
        name=name,
        prefix_mode=prefix_mode,
        prefix_source=source.name if prefix_mode == PrefixMode.FROM else None,
        started_at=started_at,
        finished_at=utc_now(),
        attempts=attempts,
    )
