# kvcache-sanity-check

Most LLM benchmarks measure throughput and time-to-first-token — they send a prompt in and check that *something* came back quickly. That means a server can score well while returning garbage. This tool checks that your inference server is returning **correct** answers, not just fast ones.

The primary target is [LMCache](https://github.com/LMCache/LMCache) with segmented prefill: when some KV cache blocks are missing and must be recomputed, this tool verifies the recomputation happened correctly and the model's answers are still accurate.

## How it works

Each test scenario loads several documents into the model's context through a multi-turn conversation, then asks a question about one of the **earlier** documents — the ones whose KV blocks were computed first and are most likely to have been evicted or incorrectly recomputed.

To get a trustworthy reference answer without spinning up a second model, the same server is called again with a unique UUID injected at the start of the system message. Because KV caches are keyed on the full token prefix, a single changed token at position 0 guarantees a complete cache miss and a clean recompute. The two answers are then compared using the model itself as a judge (also UUID-prefixed, also cache-busted).

```
┌─────────────────────────────────────────────────────────┐
│  Target call  →  answer    (may use cached blocks)      │
│  Reference call (UUID prefix)  →  answer  (clean)       │
│  Judge call (UUID prefix)  →  score + pass/fail          │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e .
```

Requires Python 3.10+.

## Quick start

```bash
# Copy and edit the sample config
cp kvcache-check.yaml.example kvcache-check.yaml
$EDITOR kvcache-check.yaml   # set target_url and model

# Run
kvcache-check
```

## Usage

```bash
# Minimal — point at a running vLLM/LMCache server
kvcache-check --target-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct

# Run each scenario 5 times to catch intermittent failures
kvcache-check --target-url http://localhost:8000 --model <model> --iterations 5

# Tighter pass threshold (default is 0.7)
kvcache-check --target-url http://localhost:8000 --model <model> --threshold 0.85

# Only catch wrong-document answers (less sensitive, good for initial testing)
kvcache-check --target-url http://localhost:8000 --model <model> --judge-prompt topic

# Use a separate, stronger model as judge
kvcache-check \
  --target-url http://localhost:8000 --model <model> \
  --judge-url https://api.openai.com --judge-model gpt-4o-mini --judge-api-key sk-...

# Save full run traces for later review
kvcache-check --target-url http://localhost:8000 --model <model> --log-file runs.jsonl

# Show both answers for every test, not just failures
kvcache-check --target-url http://localhost:8000 --model <model> --verbose
```

Exits with code 0 if all scenarios pass, 1 if any fail — suitable for CI.

A config file is auto-discovered at `./kvcache-check.yaml` or `~/.config/kvcache-check/config.yaml`. All CLI flags can be set there. See `kvcache-check.yaml.example`.

## Reviewing logs with the TUI

Run with `--log-file` to save traces, then browse them interactively:

```bash
kvcache-check --log-file runs.jsonl   # (or set log_file in config)
kvcache-logs runs.jsonl
```

The TUI shows a run list on the left and a detail pane on the right with collapsible sections for the full conversation, target answer, reference answer, and judge exchange.

```
┌─ Runs ──────────┬─ Detail ──────────────────────────────────────┐
│ early_doc_recall│  early_doc_recall  iter 1  PASS 90%            │
│   iter 1  ✓    │  ▼ Conversation                                 │
│   iter 2  ✓    │    system: You are a helpful…                   │
│   iter 3  ✗    │    user:   <History of the Internet…>           │
│ middle_doc      │    …                                            │
│   iter 1  ✓    │  ▼ Target answer                                │
│                 │  ▼ Reference answer                             │
│                 │  ▼ Judge exchange + raw response                │
└─────────────────┴────────────────────────────────────────────────┘
```

Key bindings: `j`/`k` or arrow keys to navigate, `q` to quit.

## Default scenarios

| Scenario | Mode | What it tests |
|---|---|---|
| `early_doc_recall` | multi-turn | Recall the **first** document loaded — highest eviction risk |
| `middle_doc_recall` | multi-turn | Recall the **third** document loaded — moderate eviction risk |
| `specific_fact_retrieval` | multi-turn | Retrieve a specific named fact from an early document |
| `sequential_summarize` | sequential pairs | Independent "summarize this" requests in the same chat — tests cross-turn cache contamination |

The corpus uses full Wikipedia articles (~5,000–15,000 words each) downloaded via `scripts/download_corpus.py`. Topics are distinct enough that answering about the wrong one is unambiguous.

## Judge prompts

| Prompt | Use when |
|---|---|
| `strict` (default) | Fine-grained consistency scoring — catches partial errors and detail drift |
| `topic` | Only fails if the model clearly addressed the wrong document — good for initial testing |

## Extending

**Download more corpus documents:**
```bash
python scripts/download_corpus.py "Alan Turing" "Byzantine Empire"
python scripts/download_corpus.py --list   # show defaults
```

**Add scenarios** — append to `scenarios/default.yaml` or pass `--scenarios-file path/to/custom.yaml`.

**Custom corpus** — pass `--corpus-dir /path/to/docs`. Documents must be `.txt` with `# Title` on the first line.

## Roadmap

- [ ] LMCache control API integration — programmatically set eviction/failure rate and sweep quality vs. eviction-rate curves rather than just observing
- [ ] Embedding-based similarity as an alternative to LLM-as-judge
- [ ] Structured JSON output for integration with dashboards / CI reporters
