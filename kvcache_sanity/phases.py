from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
    TestResult,
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


@dataclass
class RunState:
    """Artifacts of one run, passed from phase to phase."""

    cases: list[Case]
    reference_set: ReferenceSet | None = None
    replays: dict[str, ReplayRun] = field(default_factory=dict)
    results: list[TestResult] = field(default_factory=list)

    def require_reference_set(self) -> ReferenceSet:
        if self.reference_set is None:
            raise ValueError("no reference set: run the reference phase first")

        return self.reference_set


class Phase(ABC):
    """One step of a run: reads its inputs from the run state and stores its output there."""

    name: str

    @abstractmethod
    def run(self, state: RunState) -> None:
        ...


def _generation_status(run: RunResult, incomplete: ResultStatus) -> ResultStatus:
    if is_complete(run):
        return ResultStatus.PASS

    if run.error:
        return ResultStatus.TIMEOUT if run.timed_out else ResultStatus.ERROR

    return incomplete


def _send(context: PhaseContext, case: Case, prefix: str, label: str) -> RunResult:
    temperature = case.temperature if case.temperature is not None else context.temperature

    run = _call_api(
        context.target_client,
        context.model,
        inject_prefix(case.messages, prefix),
        context.max_tokens,
        label,
        temperature=temperature,
        empty_retries=0,
    )
    run.unique_prefix = prefix

    return run


class ReferencePhase(Phase):
    """Generate the reference of every case under a fresh cache-busting prefix (a full prefill).

    Every case gets a record, failed ones included, so later phases see the whole case set.
    The accepted prefix is the one later replays reuse to read the blocks this phase stored.
    """

    name = "reference"

    def __init__(self, context: PhaseContext) -> None:
        self.context = context

    def run(self, state: RunState) -> None:
        state.reference_set = ReferenceSet(
            created_at=utc_now(),
            model=self.context.model,
            temperature=self.context.temperature,
            max_tokens=self.context.max_tokens,
            records=[self._record(case) for case in state.cases],
        )

    def _record(self, case: Case) -> ReferenceRecord:
        label = f"reference, case={case.case_id}"

        def generate(prefix: str) -> RunResult:
            return _send(self.context, case, prefix, label)

        attempts = prepare_reference(
            generate,
            str(uuid.uuid4()),
            retries=self.context.reference_retries,
            label=label,
        )
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


class ReplayPhase(Phase):
    """Send the conversation of every case with a complete reference, without judging the answers.

    prefix_mode picks the cache-busting prefix per case (see PrefixMode); PrefixMode.FROM takes
    the prefixes of the replay named by source.

    Requests go one at a time, in rounds of one request per case. With duration, rounds repeat
    until it has elapsed (at least one), capped by iterations when that is non-zero.

    A case without a complete reference is not sent; it gets one attempt carrying the
    reference status, so no case drops out of the results.
    """

    def __init__(
        self,
        context: PhaseContext,
        name: str,
        prefix_mode: PrefixMode,
        iterations: int,
        duration: float | None = None,
        source: str | None = None,
    ) -> None:
        if prefix_mode == PrefixMode.FROM and source is None:
            raise ValueError("prefix mode 'from' needs the source replay")

        self.context = context
        self.name = name
        self.prefix_mode = prefix_mode
        self.iterations = iterations
        self.duration = duration
        self.source = source if prefix_mode == PrefixMode.FROM else None

    def run(self, state: RunState) -> None:
        reference_set = state.require_reference_set()
        records = reference_set.records
        prefixes = self._prefixes(state, reference_set)

        sendable = [record for record in records if record.status == ResultStatus.PASS]
        attempts = [self._carried_attempt(record) for record in records if record.status != ResultStatus.PASS]

        missing = [record.case.case_id for record in sendable if record.case.case_id not in prefixes]
        if missing:
            raise ValueError(f"prefix mode {self.prefix_mode.value!r}: no prefix for case(s) {', '.join(missing)}")

        started_at = utc_now()
        deadline = time.monotonic() + self.duration if self.duration is not None else None
        rounds = 0

        while sendable and self._more_rounds(rounds, deadline):
            rounds += 1

            for record in sendable:
                attempts.append(self._attempt(record, prefixes[record.case.case_id], rounds))

        order = {record.case.case_id: index for index, record in enumerate(records)}
        attempts.sort(key=lambda attempt: (order[attempt.case_id], attempt.seq))

        state.replays[self.name] = ReplayRun(
            name=self.name,
            prefix_mode=self.prefix_mode,
            prefix_source=self.source,
            started_at=started_at,
            finished_at=utc_now(),
            attempts=attempts,
        )

    def _prefixes(self, state: RunState, reference_set: ReferenceSet) -> dict[str, str]:
        """Cache-busting prefix per case_id."""

        if self.prefix_mode == PrefixMode.REUSE:
            return {record.case.case_id: record.prefix for record in reference_set.records}

        if self.prefix_mode == PrefixMode.FRESH:
            return {record.case.case_id: str(uuid.uuid4()) for record in reference_set.records}

        if self.source not in state.replays:
            raise ValueError(f"prefix mode 'from': no replay named {self.source!r}")

        return {attempt.case_id: attempt.prefix for attempt in state.replays[self.source].attempts}

    def _more_rounds(self, done: int, deadline: float | None) -> bool:
        if self.iterations and done >= self.iterations:
            return False

        if deadline is None:
            return done < self.iterations

        return done == 0 or time.monotonic() < deadline

    @staticmethod
    def _carried_attempt(record: ReferenceRecord) -> ReplayAttempt:
        return ReplayAttempt(
            case_id=record.case.case_id,
            seq=1,
            prefix=record.prefix,
            status=record.status,
            error=record.error,
        )

    def _attempt(self, record: ReferenceRecord, prefix: str, seq: int) -> ReplayAttempt:
        case = record.case

        run = _send(self.context, case, prefix, f"replay={self.name}, case={case.case_id}, seq={seq}")
        status = _generation_status(run, ResultStatus.TARGET_INCOMPLETE)

        console.print(
            f"  {escape(self.name)} [bold]{escape(case.case_id)}[/] #{seq}: {status.value} "
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
