import sys
from pathlib import Path

import click
import yaml
from openai import OpenAI
from rich.console import Console

from kvcache_sanity.corpus import load_documents
from kvcache_sanity.evaluator import evaluate_answers
from kvcache_sanity.models import EvaluationResult, Scenario, TestResult
from kvcache_sanity.runner import get_reference_answer, run_scenario
from kvcache_sanity import report

console = Console()

DEFAULT_SCENARIOS_FILE = Path(__file__).parent.parent / "scenarios" / "default.yaml"


def _load_scenarios(path: Path) -> list[Scenario]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [Scenario(**s) for s in data["scenarios"]]


@click.command()
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
@click.option("--verbose", "-v", is_flag=True,
              help="Print target and reference answers for every test, not just failures.")
def cli(
    target_url, model, api_key,
    judge_url, judge_model, judge_api_key,
    iterations, threshold, scenarios_file, corpus_dir,
    max_tokens, verbose,
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

    console.print("[bold]KV Cache Sanity Check[/]")
    console.print(f"Target : {target_base}  model={model}")
    console.print(f"Scenarios: {len(scenarios)}  iterations: {iterations}  threshold: {threshold}")
    console.print()

    all_results: list[TestResult] = []

    for scenario in scenarios:
        console.print(f"[bold cyan]{scenario.id}[/] — {scenario.description}")

        for i in range(1, iterations + 1):
            try:
                target_answer = run_scenario(
                    scenario, documents, target_client, model, max_tokens=max_tokens
                )
                reference_answer = get_reference_answer(
                    scenario, documents, target_client, model, max_tokens=max_tokens
                )
                evaluation = evaluate_answers(
                    question=scenario.question,
                    target_answer=target_answer,
                    reference_answer=reference_answer,
                    client=target_client,
                    model=judge_model_name,
                    threshold=threshold,
                    judge_client=judge_client,
                )
                result = TestResult(
                    scenario_id=scenario.id,
                    iteration=i,
                    question=scenario.question,
                    target_answer=target_answer,
                    reference_answer=reference_answer,
                    evaluation=evaluation,
                )
            except Exception as exc:
                result = TestResult(
                    scenario_id=scenario.id,
                    iteration=i,
                    question=scenario.question,
                    target_answer="",
                    reference_answer="",
                    evaluation=EvaluationResult(score=0.0, reasoning=str(exc), passed=False),
                    error=str(exc),
                )

            all_results.append(result)
            report.print_result(result, verbose=verbose)

        console.print()

    report.print_summary(all_results)

    if not all(r.evaluation.passed for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    cli()
