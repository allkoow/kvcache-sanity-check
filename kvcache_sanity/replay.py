"""Replay a specific request from a kvcache-check log file.

Looks up a log entry by target_request_id, reconstructs the exact messages
from the original scenario and corpus (using the shared UUID prefix stored in
the log), and resends the request N times — useful for reproducing intermittent
failures seen in server logs.

Usage:
    kvcache-replay runs.jsonl --request-id chatcmpl-abc123 --count 10
"""
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml
from openai import OpenAI
from rich.console import Console
from rich.rule import Rule

from kvcache_sanity.corpus import load_documents
from kvcache_sanity.logger import load_run_logs
from kvcache_sanity.models import Scenario
from kvcache_sanity.runner import _call_api, build_messages, build_pairs_messages

console = Console()

_DEFAULT_SCENARIOS_FILE = Path(__file__).parent.parent / "scenarios" / "default.yaml"


def _load_scenarios(path: Path) -> list[Scenario]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [Scenario(**s) for s in data["scenarios"]]


def _parse_scenario_id(scenario_id: str) -> tuple[str, int | None]:
    """Split 'base_id[pair N/M (doc=...)]' into (base_id, pair_idx).

    pair_idx is 0-indexed. Returns (scenario_id, None) for non-pairs scenarios.
    """
    m = re.match(r"^(.+)\[pair (\d+)/\d+", scenario_id)
    if m:
        return m.group(1), int(m.group(2)) - 1
    return scenario_id, None


@click.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--request-id", required=True, metavar="ID",
              help="target_request_id of the log entry to replay.")
@click.option("--count", default=5, show_default=True, type=int,
              help="Number of times to resend the request.")
@click.option("--target-url", required=True,
              help="Base URL of the inference server (e.g. http://localhost:8000).")
@click.option("--model", required=True,
              help="Model name as registered on the server.")
@click.option("--api-key", default="EMPTY", show_default=True,
              help="API key for the server.")
@click.option("--max-tokens", default=1024, show_default=True, type=int,
              help="Max tokens for each replayed response.")
@click.option("--temperature", default=0.1, show_default=True, type=float,
              help="Sampling temperature used if the log entry does not record one.")
@click.option("--scenarios-file", default=None, type=click.Path(exists=True),
              help="Scenarios YAML file. Defaults to scenarios/default.yaml.")
@click.option("--corpus-dir", default=None, type=click.Path(exists=True),
              help="Corpus directory. Defaults to the bundled corpus/.")
def replay(
    log_file, request_id, count,
    target_url, model, api_key, max_tokens, temperature,
    scenarios_file, corpus_dir,
) -> None:
    """Replay a logged request N times to reproduce or characterise a failure.

    Reconstructs the exact conversation from the original scenario and corpus
    using the UUID prefix stored in the log, then sends it COUNT times and
    prints each response with its request_id and token count.
    """
    # --- find the log entry ---
    logs = load_run_logs(Path(log_file))
    entry = next((l for l in logs if l.target_request_id == request_id), None)
    if entry is None:
        console.print(f"[red]No log entry found with target_request_id={request_id!r}[/]")
        sys.exit(1)

    if entry.temperature is not None:
        effective_temperature = entry.temperature
    else:
        console.print(
            f"[yellow]WARNING[/yellow]: log entry has no temperature recorded; "
            f"using --temperature={temperature}"
        )
        effective_temperature = temperature

    console.print(f"Found: [bold]{entry.scenario_id}[/]  iter {entry.iteration}  "
                  f"original @ {entry.target_request_time or entry.timestamp}")
    console.print(f"UUID prefix: {entry.reference_prefix}")
    console.print(f"Temperature: {effective_temperature}")
    console.print()

    # --- reconstruct messages ---
    base_id, pair_idx = _parse_scenario_id(entry.scenario_id)

    scenarios_path = Path(scenarios_file) if scenarios_file else _DEFAULT_SCENARIOS_FILE
    scenarios = _load_scenarios(scenarios_path)
    scenario = next((s for s in scenarios if s.id == base_id), None)
    if scenario is None:
        console.print(f"[red]Scenario {base_id!r} not found in {scenarios_path}[/]")
        sys.exit(1)

    corpus_path = Path(corpus_dir) if corpus_dir else None
    documents = load_documents(corpus_path)

    if pair_idx is not None:
        messages = build_pairs_messages(
            scenario.pairs, documents, pair_idx, entry.reference_prefix
        )
    else:
        messages = build_messages(scenario, documents, entry.reference_prefix)

    # --- replay ---
    client = OpenAI(base_url=f"{target_url.rstrip('/')}/v1", api_key=api_key)

    console.print(Rule(f"Replaying {count}x"))

    passed = 0
    for i in range(1, count + 1):
        answer, finish_reason, completion_tokens, req_id, req_time = _call_api(
            client, model, messages, max_tokens,
            label=f"replay {i}/{count}",
            temperature=effective_temperature,
        )
        status = "[green]ok[/]" if answer.strip() else "[red]empty[/]"
        console.print(
            f"[{i}/{count}] {status}  "
            f"request_id={req_id}  "
            f"finish_reason={finish_reason}  "
            f"tokens={completion_tokens}  "
            f"@ {req_time}"
        )
        if answer.strip():
            console.print(f"  [dim]{answer[:300].replace(chr(10), ' ')}[/dim]")
            passed += 1

    console.print()
    console.print(Rule())
    color = "green" if passed == count else ("yellow" if passed > 0 else "red")
    console.print(f"[bold {color}]{passed}/{count} non-empty responses[/]")
