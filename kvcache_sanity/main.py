import sys
from pathlib import Path

import click
import yaml
from openai import OpenAI
from rich.console import Console

from kvcache_sanity.corpus import load_documents
from kvcache_sanity.evaluator import evaluate_answers, DEFAULT_PROMPT, PROMPT_NAMES
from kvcache_sanity.logger import RunLogger
from kvcache_sanity.models import EvaluationResult, EvaluationTrace, RunResult, Scenario, TestResult
from kvcache_sanity.runner import (
    get_reference_answer,
    get_reference_pair_answer,
    run_scenario,
    run_sequential_pair,
)
from kvcache_sanity import report

console = Console()

DEFAULT_SCENARIOS_FILE = Path(__file__).parent.parent / "scenarios" / "default.yaml"

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


def _load_scenarios(path: Path) -> list[Scenario]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [Scenario(**s) for s in data["scenarios"]]


def _run_single(
    *, scenario, documents, target_client, model, judge_model_name, judge_client,
    threshold, max_tokens, judge_prompt, iteration, all_results, run_logger, verbose,
) -> None:
    error: str | None = None
    target_run = RunResult(answer="")
    reference_run = RunResult(answer="")
    trace = EvaluationTrace(result=EvaluationResult(score=0.0, reasoning="", passed=False))

    try:
        target_run = run_scenario(scenario, documents, target_client, model, max_tokens=max_tokens)
        reference_run = get_reference_answer(scenario, documents, target_client, model, max_tokens=max_tokens)
        trace = evaluate_answers(
            question=scenario.question,
            target_answer=target_run.answer,
            reference_answer=reference_run.answer,
            client=target_client,
            model=judge_model_name,
            threshold=threshold,
            judge_client=judge_client,
            judge_prompt=judge_prompt,
        )
        result = TestResult(
            scenario_id=scenario.id, iteration=iteration, question=scenario.question,
            target_answer=target_run.answer, reference_answer=reference_run.answer,
            evaluation=trace.result,
        )
    except Exception as exc:
        error = str(exc)
        result = TestResult(
            scenario_id=scenario.id, iteration=iteration, question=scenario.question,
            target_answer=target_run.answer, reference_answer=reference_run.answer,
            evaluation=EvaluationResult(score=0.0, reasoning=str(exc), passed=False),
            error=error,
        )

    all_results.append(result)
    report.print_result(result, verbose=verbose)
    if run_logger:
        run_logger.log(scenario=scenario, iteration=iteration,
                       target=target_run, reference=reference_run, trace=trace, error=error)


def _run_sequential_pairs(
    *, scenario, documents, target_client, model, judge_model_name, judge_client,
    threshold, max_tokens, judge_prompt, iteration, all_results, run_logger, verbose,
) -> None:
    pairs = scenario.pairs
    for pair_idx in range(scenario.evaluate_from_pair, len(pairs)):
        pair = pairs[pair_idx]
        error: str | None = None
        target_run = RunResult(answer="")
        reference_run = RunResult(answer="")
        trace = EvaluationTrace(result=EvaluationResult(score=0.0, reasoning="", passed=False))

        pair_label = f"pair {pair_idx + 1}/{len(pairs)} (doc={pair.doc_id})"
        try:
            target_run = run_sequential_pair(
                pairs, documents, target_client, model, pair_idx, max_tokens=max_tokens
            )
            reference_run = get_reference_pair_answer(
                pairs, documents, target_client, model, pair_idx, max_tokens=max_tokens
            )
            trace = evaluate_answers(
                question=pair.question,
                target_answer=target_run.answer,
                reference_answer=reference_run.answer,
                client=target_client,
                model=judge_model_name,
                threshold=threshold,
                judge_client=judge_client,
                judge_prompt=judge_prompt,
            )
            result = TestResult(
                scenario_id=f"{scenario.id}[{pair_label}]",
                iteration=iteration, question=pair.question,
                target_answer=target_run.answer, reference_answer=reference_run.answer,
                evaluation=trace.result,
            )
        except Exception as exc:
            error = str(exc)
            result = TestResult(
                scenario_id=f"{scenario.id}[{pair_label}]",
                iteration=iteration, question=pair.question,
                target_answer=target_run.answer, reference_answer=reference_run.answer,
                evaluation=EvaluationResult(score=0.0, reasoning=str(exc), passed=False),
                error=error,
            )

        all_results.append(result)
        report.print_result(result, verbose=verbose)
        if run_logger:
            run_logger.log(scenario=scenario, iteration=iteration,
                           target=target_run, reference=reference_run, trace=trace, error=error)


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
@click.option("--iterations", default=1, show_default=True, type=int,
              help="Number of times to run each scenario. "
                   "Multiple iterations help catch flaky caching failures.")
@click.option("--threshold", default=0.7, show_default=True, type=float,
              help="Minimum similarity score (0.0–1.0) to count as PASS.")
@click.option("--scenarios-file", default=None, type=click.Path(exists=True),
              help="Path to a YAML file with test scenarios. Defaults to scenarios/default.yaml.")
@click.option("--corpus-dir", default=None, type=click.Path(exists=True),
              help="Directory of .txt document files. Defaults to the bundled corpus/.")
@click.option("--max-tokens", default=512, show_default=True, type=int,
              help="Max tokens for model answer responses.")
@click.option("--judge-prompt", default=DEFAULT_PROMPT, show_default=True,
              type=click.Choice(PROMPT_NAMES),
              help="Judge prompt style: 'strict' scores fine-grained consistency; "
                   "'topic' only catches wrong-document answers.")
@click.option("--log-file", default=None, type=click.Path(), metavar="PATH",
              help="Append full run traces (JSON Lines) to this file for later review with kvcache-logs.")
@click.option("--verbose", "-v", is_flag=True,
              help="Print target and reference answers for every test, not just failures.")
def cli(
    target_url, model, api_key,
    judge_url, judge_model, judge_api_key,
    iterations, threshold, scenarios_file, corpus_dir,
    max_tokens, judge_prompt, log_file, verbose,
):
    """Sanity-check LLM output correctness when using offloaded KV cache.

    Builds multi-turn conversations that load several documents into context,
    then asks questions whose answers require recalling specific earlier documents
    (the most likely victims of KV block eviction). Compares the server's answer
    against a reference obtained by forcing a full recompute via a unique cache-busting
    prefix injected into the system message.
    """
    corpus_path = Path(corpus_dir) if corpus_dir else None
    documents = load_documents(corpus_path)

    scenarios_path = Path(scenarios_file) if scenarios_file else DEFAULT_SCENARIOS_FILE
    scenarios = _load_scenarios(scenarios_path)

    # Normalise base URLs — strip trailing slash
    target_base = target_url.rstrip("/")
    target_client = OpenAI(base_url=f"{target_base}/v1", api_key=api_key)

    judge_client = None
    if judge_url:
        judge_client = OpenAI(base_url=f"{judge_url.rstrip('/')}/v1", api_key=judge_api_key)
    judge_model_name = judge_model or model

    run_logger = RunLogger(Path(log_file)) if log_file else None


    console.print("[bold]KV Cache Sanity Check[/]")
    console.print(f"Target : {target_base}  model={model}")
    console.print(f"Scenarios: {len(scenarios)}  iterations: {iterations}  threshold: {threshold}")
    if run_logger:
        console.print(f"Logging to: {log_file}")
    console.print()

    all_results: list[TestResult] = []

    for scenario in scenarios:
        console.print(f"[bold cyan]{scenario.id}[/] — {scenario.description}")

        for i in range(1, iterations + 1):
            common = dict(
                scenario=scenario, documents=documents,
                target_client=target_client, model=model,
                judge_model_name=judge_model_name, judge_client=judge_client,
                threshold=threshold, max_tokens=max_tokens, judge_prompt=judge_prompt,
                iteration=i, all_results=all_results, run_logger=run_logger, verbose=verbose,
            )
            if scenario.mode == "sequential_pairs":
                _run_sequential_pairs(**common)
            else:
                _run_single(**common)

        console.print()

    report.print_summary(all_results)

    if not all(r.evaluation.passed for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    cli()
