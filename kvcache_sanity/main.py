from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import click
import yaml
from openai import OpenAI
from rich.console import Console

from kvcache_sanity.corpus import load_documents
from kvcache_sanity.evaluator import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_RETRIES,
    DEFAULT_PROMPT,
    PROMPT_NAMES,
    JudgeConfig,
    evaluate_answers,
)
from kvcache_sanity.logger import RunLogger
from kvcache_sanity.models import (
    Document,
    EvaluationResult,
    EvaluationTrace,
    ResultStatus,
    RunResult,
    Scenario,
    ScenarioPair,
    TestResult,
)
from kvcache_sanity.runner import (
    DEFAULT_REFERENCE_RETRIES,
    describe_generation,
    get_reference_answer,
    get_reference_pair_answer,
    is_complete,
    prepare_reference,
    run_scenario,
    run_sequential_pair,
)
from kvcache_sanity.scenarios import find_named_scenarios, load_scenarios
from kvcache_sanity import report

console = Console()

_CONFIG_SEARCH_PATHS = [
    Path("kvcache-check.yaml"),
    Path.home() / ".config" / "kvcache-check" / "config.yaml",
]


def _load_config(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Normalise keys: YAML may use dashes or underscores; Click wants underscores.
    return {k.replace("-", "_"): v for k, v in data.items()}


def _find_config(explicit: str | None) -> dict:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return _load_config(p)
        raise click.BadParameter(f"Config file not found: {explicit}", param_hint="--config")
    for p in _CONFIG_SEARCH_PATHS:
        if p.exists():
            return _load_config(p)
    return {}


class _ConfigFileCommand(click.Command):
    """Command subclass that injects a YAML config file as Click's default_map."""

    def make_context(self, info_name, args, **kwargs):
        # Peek at args to find --config value before Click processes anything.
        config_path: str | None = None
        for i, arg in enumerate(args):
            if arg in ("--config", "-c") and i + 1 < len(args):
                config_path = args[i + 1]
                break
            if arg.startswith("--config="):
                config_path = arg.split("=", 1)[1]
                break
        try:
            config = _find_config(config_path)
        except click.BadParameter as exc:
            raise exc
        if config:
            kwargs.setdefault("default_map", {}).update(config)
        return super().make_context(info_name, args, **kwargs)


@dataclass(frozen=True)
class RunContext:
    """Settings and sinks shared by every scenario of a run."""

    documents: dict[str, Document]
    target_client: OpenAI
    model: str
    judge: JudgeConfig
    max_tokens: int
    iterations: int
    reference_retries: int
    all_results: list[TestResult]
    run_logger: RunLogger | None
    verbose: bool


@dataclass(frozen=True)
class ReferenceRun:
    """Every reference attempt for one evaluated question, or the error that stopped them."""

    attempts: list[RunResult] = field(default_factory=list)
    error: str | None = None

    @property
    def run(self) -> RunResult:
        """The reference to compare against; incomplete when every attempt failed."""
        return self.attempts[-1] if self.attempts else RunResult(answer="")


def _pair_label(pairs: list[ScenarioPair], pair_index: int) -> str:
    return f"pair {pair_index + 1}/{len(pairs)} (doc={pairs[pair_index].doc_id})"


def _run_reference(
    context: RunContext,
    scenario: Scenario,
    temperature: float,
    prefix: str,
    pair_index: int | None = None,
) -> ReferenceRun:
    """Prepare the reference for a scenario, or for one of its pairs in sequential_pairs mode."""
    if pair_index is None:
        label = f"scenario={scenario.id}"

        def generate(attempt_prefix: str) -> RunResult:
            return get_reference_answer(
                scenario, context.documents, context.target_client, context.model,
                max_tokens=context.max_tokens, prefix=attempt_prefix, temperature=temperature,
            )
    else:
        label = f"scenario={scenario.id}, {_pair_label(scenario.pairs, pair_index)}"

        def generate(attempt_prefix: str) -> RunResult:
            return get_reference_pair_answer(
                scenario.pairs, context.documents, context.target_client, context.model, pair_index,
                max_tokens=context.max_tokens, prefix=attempt_prefix, temperature=temperature,
            )

    try:
        attempts = prepare_reference(generate, prefix, retries=context.reference_retries, label=label)
    except Exception as exc:
        return ReferenceRun(error=str(exc) or repr(exc))

    return ReferenceRun(attempts=attempts)


def _evaluation_status(trace: EvaluationTrace) -> ResultStatus:
    if trace.judge_error is not None:
        return ResultStatus.JUDGE_ERROR
    return ResultStatus.PASS if trace.result.passed else ResultStatus.FAIL


def _incomplete_target_evaluation(target_run: RunResult) -> EvaluationResult:
    reasoning = f"[target incomplete, judge skipped] {describe_generation(target_run)}"
    return EvaluationResult(score=0.0, reasoning=reasoning, passed=False)


def _incomplete_reference_evaluation(attempts: list[RunResult]) -> EvaluationResult:
    reasoning = f"[reference incomplete after {len(attempts)} attempt(s)] {describe_generation(attempts[-1])}"
    return EvaluationResult(score=0.0, reasoning=reasoning, passed=False)


def _run_scenario_iterations(context: RunContext, scenario: Scenario, temperature: float) -> None:
    # Reference runs first with a shared UUID, warming the KV cache.
    # All target iterations reuse the UUID of the accepted reference so they hit
    # blocks cached by the reference call — every iteration exercises the cache equally.
    reference = _run_reference(context, scenario, temperature, str(uuid.uuid4()))
    reference_run = reference.run
    ref_error = reference.error

    for i in range(1, context.iterations + 1):
        error = ref_error
        target_run = RunResult(answer="")
        trace = EvaluationTrace(result=EvaluationResult(score=0.0, reasoning="", passed=False))

        if ref_error:
            result = TestResult(
                scenario_id=scenario.id, iteration=i, question=scenario.question,
                target_answer="", reference_answer="",
                evaluation=EvaluationResult(score=0.0, reasoning=ref_error, passed=False),
                status=ResultStatus.ERROR,
                error=ref_error,
            )
        elif not is_complete(reference_run):
            trace = EvaluationTrace(result=_incomplete_reference_evaluation(reference.attempts))
            result = TestResult(
                scenario_id=scenario.id, iteration=i, question=scenario.question,
                target_answer="", reference_answer=reference_run.answer,
                evaluation=trace.result,
                status=ResultStatus.REFERENCE_INCOMPLETE,
            )
        else:
            try:
                target_run = run_scenario(
                    scenario, context.documents, context.target_client, context.model,
                    unique_prefix=reference_run.unique_prefix, max_tokens=context.max_tokens,
                    temperature=temperature, empty_retries=0,
                )
                if is_complete(target_run):
                    trace = evaluate_answers(scenario.question, target_run.answer, reference_run.answer, context.judge)
                    status = _evaluation_status(trace)
                else:
                    trace = EvaluationTrace(result=_incomplete_target_evaluation(target_run))
                    status = ResultStatus.TARGET_INCOMPLETE
                result = TestResult(
                    scenario_id=scenario.id, iteration=i, question=scenario.question,
                    target_answer=target_run.answer, reference_answer=reference_run.answer,
                    evaluation=trace.result,
                    status=status,
                    target_request_id=target_run.request_id,
                    target_truncation=target_run.truncation,
                )
            except Exception as exc:
                error = str(exc)
                result = TestResult(
                    scenario_id=scenario.id, iteration=i, question=scenario.question,
                    target_answer=target_run.answer, reference_answer=reference_run.answer,
                    evaluation=EvaluationResult(score=0.0, reasoning=str(exc), passed=False),
                    status=ResultStatus.ERROR,
                    error=error,
                )

        context.all_results.append(result)
        report.print_result(result, verbose=context.verbose)
        if context.run_logger:
            context.run_logger.log(scenario=scenario, iteration=i,
                                   target=target_run, reference=reference_run, trace=trace, error=error,
                                   temperature=temperature, status=result.status,
                                   reference_attempts=reference.attempts)


def _run_sequential_pairs(context: RunContext, scenario: Scenario, temperature: float) -> None:
    pairs = scenario.pairs
    # Reference runs first with a shared UUID per scenario, warming the KV cache.
    # All target iterations reuse the same UUID so every iteration exercises the cache.
    shared_prefix = str(uuid.uuid4())

    for pair_idx in range(scenario.evaluate_from_pair, len(pairs)):
        pair = pairs[pair_idx]
        pair_label = _pair_label(pairs, pair_idx)

        reference = _run_reference(context, scenario, temperature, shared_prefix, pair_index=pair_idx)
        reference_run = reference.run
        ref_error = reference.error

        for i in range(1, context.iterations + 1):
            error = ref_error
            target_run = RunResult(answer="")
            trace = EvaluationTrace(result=EvaluationResult(score=0.0, reasoning="", passed=False))

            if ref_error:
                result = TestResult(
                    scenario_id=f"{scenario.id}[{pair_label}]",
                    iteration=i, question=pair.question,
                    target_answer="", reference_answer="",
                    evaluation=EvaluationResult(score=0.0, reasoning=ref_error, passed=False),
                    status=ResultStatus.ERROR,
                    error=ref_error,
                )
            elif not is_complete(reference_run):
                trace = EvaluationTrace(result=_incomplete_reference_evaluation(reference.attempts))
                result = TestResult(
                    scenario_id=f"{scenario.id}[{pair_label}]",
                    iteration=i, question=pair.question,
                    target_answer="", reference_answer=reference_run.answer,
                    evaluation=trace.result,
                    status=ResultStatus.REFERENCE_INCOMPLETE,
                )
            else:
                try:
                    target_run = run_sequential_pair(
                        pairs, context.documents, context.target_client, context.model, pair_idx,
                        unique_prefix=reference_run.unique_prefix, max_tokens=context.max_tokens,
                        temperature=temperature, empty_retries=0,
                    )
                    if is_complete(target_run):
                        trace = evaluate_answers(pair.question, target_run.answer, reference_run.answer, context.judge)
                        status = _evaluation_status(trace)
                    else:
                        trace = EvaluationTrace(result=_incomplete_target_evaluation(target_run))
                        status = ResultStatus.TARGET_INCOMPLETE
                    result = TestResult(
                        scenario_id=f"{scenario.id}[{pair_label}]",
                        iteration=i, question=pair.question,
                        target_answer=target_run.answer, reference_answer=reference_run.answer,
                        evaluation=trace.result,
                        status=status,
                        target_request_id=target_run.request_id,
                        target_truncation=target_run.truncation,
                    )
                except Exception as exc:
                    error = str(exc)
                    result = TestResult(
                        scenario_id=f"{scenario.id}[{pair_label}]",
                        iteration=i, question=pair.question,
                        target_answer=target_run.answer, reference_answer=reference_run.answer,
                        evaluation=EvaluationResult(score=0.0, reasoning=str(exc), passed=False),
                        status=ResultStatus.ERROR,
                        error=error,
                    )

            context.all_results.append(result)
            report.print_result(result, verbose=context.verbose)
            if context.run_logger:
                context.run_logger.log(scenario=scenario, iteration=i,
                                       target=target_run, reference=reference_run, trace=trace, error=error,
                                       temperature=temperature, status=result.status,
                                       reference_attempts=reference.attempts)


@click.command(cls=_ConfigFileCommand)
@click.option("--config", "-c", default=None, is_eager=True, expose_value=False,
              type=click.Path(), metavar="PATH",
              help="YAML config file. Defaults: ./kvcache-check.yaml, "
                   "~/.config/kvcache-check/config.yaml.")
@click.option("--target-url", required=True,
              help="Base URL of the target inference server (OpenAI-compatible, e.g. http://localhost:8000).")
@click.option("--model", required=True,
              help="Model name as registered on the server.")
@click.option("--api-key", default="EMPTY", show_default=True,
              help="API key for the target server. Use 'EMPTY' for local servers that don't check it.")
@click.option("--judge-url", default=None,
              help="Optional separate endpoint for the LLM-as-judge evaluation call. "
                   "Defaults to --target-url (evaluation still uses a unique prefix to bust the cache).")
@click.option("--judge-model", default=None,
              help="Model name for the judge. Defaults to --model.")
@click.option("--judge-api-key", default="EMPTY", show_default=True,
              help="API key for the judge endpoint.")
@click.option("--temperature", default=0.1, show_default=True, type=float,
              help="Sampling temperature for scenario runs (0.0 = deterministic). "
                   "Per-scenario YAML 'temperature' field overrides this.")
@click.option("--judge-temperature", default=0.0, show_default=True, type=float,
              help="Sampling temperature for the LLM-as-judge evaluation call.")
@click.option("--judge-max-tokens", default=DEFAULT_JUDGE_MAX_TOKENS, show_default=True, type=int,
              help="Max tokens for the judge response. Reasoning models spend part of it before the JSON verdict.")
@click.option("--judge-retries", default=DEFAULT_JUDGE_RETRIES, show_default=True, type=click.IntRange(min=0),
              help="Extra judge calls when a response carries no valid verdict (empty, truncated, not JSON or "
                   "wrong schema). A valid verdict, including FAIL, is never retried; when all attempts fail the "
                   "case is JUDGE_ERROR.")
@click.option("--reference-retries", default=DEFAULT_REFERENCE_RETRIES, show_default=True,
              type=click.IntRange(min=0),
              help="Extra reference generations, each under a fresh cache-busting prefix, when the reference "
                   "is truncated or empty. Target answers are never retried.")
@click.option("--iterations", default=1, show_default=True, type=int,
              help="Number of times to run each scenario. "
                   "Multiple iterations help catch flaky caching failures.")
@click.option("--threshold", default=0.7, show_default=True, type=float,
              help="Minimum similarity score (0.0–1.0) to count as PASS.")
@click.option("--scenarios", "scenarios_name", default=None, metavar="NAME",
              help="Named scenario set, e.g. '32k' — resolved against the user scenarios dir "
                   "then the bundled presets/default.yaml (user wins on a name collision). "
                   "Mutually exclusive with --scenarios-file.")
@click.option("--scenarios-file", default=None, type=click.Path(exists=True),
              help="Path to a YAML file with test scenarios, used exclusively. Without this or "
                   "--scenarios, loads the bundled default.yaml merged with any *.yaml files in "
                   "the user scenarios dir (see paths.user_data_dir(); user scenarios win on a "
                   "matching id).")
@click.option("--corpus-dir", default=None, type=click.Path(exists=True),
              help="Directory of .txt document files, used exclusively. Without this, loads the "
                   "bundled corpus merged with the user corpus dir (see paths.user_data_dir(); "
                   "user documents win on a matching doc_id).")
@click.option("--max-tokens", default=1024, show_default=True, type=int,
              help="Max tokens for model answer responses.")
@click.option("--judge-prompt", default=DEFAULT_PROMPT, show_default=True,
              type=click.Choice(PROMPT_NAMES),
              help="Judge prompt style: 'strict' scores fine-grained consistency; "
                   "'topic' only catches wrong-document answers.")
@click.option("--log-file", default=None, type=click.Path(), metavar="PATH",
              help="Append full run traces (JSON Lines) to this file for later review with kvcache-logs.")
@click.option("--accuracy-csv", default=None, type=click.Path(), metavar="PATH",
              help="Write a CloudAI-compatible accuracy CSV (Task,Correct,Total,Accuracy,Invalid,"
                   "ReferenceIncomplete,TargetIncomplete,JudgeError,TargetLoop with an OVERALL row) to PATH. "
                   "Per-scenario pass/fail is aggregated into a 0.0-1.0 fraction; Invalid and TargetIncomplete "
                   "iterations count in Total; TargetLoop is the subset of TargetIncomplete cut while repeating. "
                   "The file is written regardless of the exit code.")
@click.option("--verbose", "-v", is_flag=True,
              help="Print target and reference answers for every test, not just failures.")
def cli(
    target_url, model, api_key,
    judge_url, judge_model, judge_api_key,
    temperature, judge_temperature, judge_max_tokens, judge_retries, reference_retries,
    iterations, threshold, scenarios_name, scenarios_file, corpus_dir,
    max_tokens, judge_prompt, log_file, accuracy_csv, verbose,
):
    """Sanity-check LLM output correctness when using offloaded KV cache.

    Builds multi-turn conversations that load several documents into context,
    then asks questions whose answers require recalling specific earlier documents
    (the most likely victims of KV block eviction). Compares the server's answer
    against a reference obtained by forcing a full recompute via a unique cache-busting
    prefix injected into the system message.
    """
    if scenarios_name and scenarios_file:
        raise click.UsageError("--scenarios and --scenarios-file are mutually exclusive.")

    corpus_path = Path(corpus_dir) if corpus_dir else None
    documents = load_documents(corpus_path)

    if scenarios_name:
        try:
            scenarios_file = find_named_scenarios(scenarios_name)
        except FileNotFoundError as exc:
            raise click.BadParameter(str(exc), param_hint="--scenarios")
    scenarios = load_scenarios(scenarios_file)

    # Normalise base URLs — strip trailing slash
    target_base = target_url.rstrip("/")
    target_client = OpenAI(base_url=f"{target_base}/v1", api_key=api_key)

    judge_client = target_client

    if judge_url:
        judge_client = OpenAI(base_url=f"{judge_url.rstrip('/')}/v1", api_key=judge_api_key)

    judge = JudgeConfig(
        client=judge_client,
        model=judge_model or model,
        prompt=judge_prompt,
        temperature=judge_temperature,
        max_tokens=judge_max_tokens,
        retries=judge_retries,
        threshold=threshold,
    )

    run_logger = RunLogger(Path(log_file)) if log_file else None

    console.print("[bold]KV Cache Sanity Check[/]")
    console.print(f"Target : {target_base}  model={model}")
    console.print(f"Scenarios: {len(scenarios)}  iterations: {iterations}  threshold: {threshold}")
    console.print(f"Temperature: {temperature}  Judge temperature: {judge_temperature}")
    console.print(f"Judge max tokens: {judge_max_tokens}  Judge retries: {judge_retries}  "
                  f"Reference retries: {reference_retries}")
    if run_logger:
        console.print(f"Logging to: {log_file}")
    console.print()

    all_results: list[TestResult] = []
    context = RunContext(
        documents=documents,
        target_client=target_client,
        model=model,
        judge=judge,
        max_tokens=max_tokens,
        iterations=iterations,
        reference_retries=reference_retries,
        all_results=all_results,
        run_logger=run_logger,
        verbose=verbose,
    )

    for scenario in scenarios:
        console.print(f"[bold cyan]{scenario.id}[/] — {scenario.description}")

        effective_temp = scenario.temperature if scenario.temperature is not None else temperature

        if scenario.mode == "sequential_pairs":
            _run_sequential_pairs(context, scenario, effective_temp)
        else:
            _run_scenario_iterations(context, scenario, effective_temp)

        console.print()

    report.print_summary(all_results)

    if accuracy_csv:
        report.write_accuracy_csv(all_results, Path(accuracy_csv))
        console.print(f"Accuracy CSV written to: {accuracy_csv}")

    if not all(r.evaluation.passed for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    cli()
