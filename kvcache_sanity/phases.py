from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import OpenAI
from rich.console import Console
from rich.markup import escape

from kvcache_sanity import report
from kvcache_sanity.evaluator import JudgeConfig, evaluate_answers, verdict_status
from kvcache_sanity.logger import reference_attempt_log
from kvcache_sanity.models import (
    Case,
    EvaluationResult,
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
    """One step of a run: reads its inputs from the run state and stores its output there.

    A phase interrupted by Ctrl+C or SIGTERM leaves in the state what it finished so far.
    A phase with runs_after_interrupt still runs after an earlier phase was interrupted.
    """

    name: str
    runs_after_interrupt: bool = False

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

    Every request reuses the prefix of the case's reference, so it reads the blocks the
    reference phase stored.

    Requests go one at a time, in rounds of one request per case. With duration, rounds repeat
    until it has elapsed (at least one), capped by iterations when that is non-zero.

    A case without a complete reference is not sent; it gets one attempt carrying the
    reference status, so no case drops out of the results.
    """

    def __init__(
        self,
        context: PhaseContext,
        name: str,
        iterations: int,
        duration: float | None = None,
    ) -> None:
        self.context = context
        self.name = name
        self.iterations = iterations
        self.duration = duration

    def run(self, state: RunState) -> None:
        records = state.require_reference_set().records
        replay = ReplayRun(name=self.name, started_at=utc_now())
        state.replays[self.name] = replay

        sendable = [record for record in records if record.status == ResultStatus.PASS]
        replay.attempts = [self._carried_attempt(record) for record in records if record.status != ResultStatus.PASS]

        deadline = time.monotonic() + self.duration if self.duration is not None else None
        rounds = 0

        try:
            while sendable and self._more_rounds(rounds, deadline):
                rounds += 1

                for record in sendable:
                    replay.attempts.append(self._attempt(record, rounds))

            replay.finished_at = utc_now()
        finally:
            order = {record.case.case_id: index for index, record in enumerate(records)}
            replay.attempts.sort(key=lambda attempt: (order[attempt.case_id], attempt.seq))

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

    def _attempt(self, record: ReferenceRecord, seq: int) -> ReplayAttempt:
        case = record.case

        run = _send(self.context, case, record.prefix, f"replay={self.name}, case={case.case_id}, seq={seq}")
        status = _generation_status(run, ResultStatus.TARGET_INCOMPLETE)

        console.print(
            f"  {escape(self.name)} [bold]{escape(case.case_id)}[/] #{seq}: {status.value} "
            f"({escape(describe_generation(run))})"
        )

        return ReplayAttempt(
            case_id=case.case_id,
            seq=seq,
            prefix=record.prefix,
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


class EvaluatePhase(Phase):
    """Judge every attempt of every replay against the reference answer of its case.

    Only PASS attempts are judged. Any other attempt keeps its status (a failed request,
    a truncated answer, a case without a complete reference) and is never sent to the judge.
    Results come in replay order, each tagged with its replay name as phase.

    It also runs after an interrupt, so the attempts that came in before it still count.
    on_result gets every result as soon as it is known, e.g. to keep the accuracy CSV current.
    """

    name = "evaluate"
    runs_after_interrupt = True

    def __init__(
        self,
        judge: JudgeConfig,
        on_result: Callable[[TestResult], None] | None = None,
        verbose: bool = False,
    ) -> None:
        self.judge = judge
        self.on_result = on_result
        self.verbose = verbose

    def run(self, state: RunState) -> None:
        if state.reference_set is None:
            return

        references = {record.case.case_id: record for record in state.reference_set.records}

        for replay in state.replays.values():
            for attempt in replay.attempts:
                result = self._result(replay.name, references[attempt.case_id], attempt)
                state.results.append(result)

                if self.on_result:
                    self.on_result(result)

    def _result(self, phase: str, record: ReferenceRecord, attempt: ReplayAttempt) -> TestResult:
        case = record.case

        if attempt.status == ResultStatus.PASS:
            trace = evaluate_answers(case.question, attempt.answer, record.answer, self.judge)
            evaluation = trace.result
            status = verdict_status(trace)
        else:
            reasoning = attempt.error or f"[{attempt.status.value}, judge skipped]"
            evaluation = EvaluationResult(score=0.0, reasoning=reasoning, passed=False)
            status = attempt.status

        result = TestResult(
            scenario_id=case.scenario_id,
            iteration=attempt.seq,
            question=case.question,
            target_answer=attempt.answer,
            reference_answer=record.answer,
            evaluation=evaluation,
            status=status,
            error=attempt.error,
            target_request_id=attempt.request_id,
            target_truncation=attempt.truncation,
            phase=phase,
        )

        console.print(f"  {escape(phase)} [bold]{escape(case.case_id)}[/]:", end="")
        report.print_result(result, verbose=self.verbose)

        return result
