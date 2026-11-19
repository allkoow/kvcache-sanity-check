from __future__ import annotations

import uuid
from dataclasses import dataclass

from openai import OpenAI
from rich.console import Console
from rich.markup import escape

from kvcache_sanity.logger import reference_attempt_log
from kvcache_sanity.models import Case, ReferenceRecord, ReferenceSet, ResultStatus, RunResult
from kvcache_sanity.runner import _call_api, describe_generation, inject_prefix, is_complete, prepare_reference, utc_now

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
    return case.temperature if case.temperature is not None else context.temperature


def _reference_status(run: RunResult) -> ResultStatus:
    if is_complete(run):
        return ResultStatus.PASS
    if run.error:
        return ResultStatus.TIMEOUT if run.timed_out else ResultStatus.ERROR
    return ResultStatus.REFERENCE_INCOMPLETE


def _reference_record(context: PhaseContext, case: Case) -> ReferenceRecord:
    label = f"reference, case={case.case_id}"

    def generate(prefix: str) -> RunResult:
        run = _call_api(
            context.target_client, context.model, inject_prefix(case.messages, prefix), context.max_tokens,
            label, temperature=_case_temperature(context, case), empty_retries=0,
        )
        run.unique_prefix = prefix
        return run

    attempts = prepare_reference(generate, str(uuid.uuid4()), retries=context.reference_retries, label=label)
    accepted = attempts[-1]
    status = _reference_status(accepted)
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
