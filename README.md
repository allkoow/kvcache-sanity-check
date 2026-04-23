# kvcache-sanity-check

Most LLM benchmarks measure throughput and time-to-first-token — they send a prompt in and check that *something* came back quickly. That means a server can score well while returning garbage. This tool checks that your inference server is returning **correct** answers, not just fast ones.

The primary target is [LMCache](https://github.com/LMCache/LMCache) with segmented prefill: when some KV cache blocks are missing and must be recomputed, this tool verifies the recomputation happened correctly and the model's answers are still accurate.

## How it works

Each test scenario loads several documents into the model's context through a multi-turn conversation, then asks a question that requires recalling a specific document. Scenarios are designed so the question targets a document whose answer is unambiguous — answering from the wrong document is obvious.

For each scenario, the tool generates a single stable UUID (the *target prefix*) and runs the scenario multiple times using that prefix. Because KV caches are keyed on the full token prefix:

- **Iteration 1** is a cold start — no blocks are cached yet for this prefix. Any failure here is a server-side bug unrelated to caching.
- **Iteration 2+** can hit KV blocks cached by iteration 1, which LMCache may have offloaded to CPU/disk. These iterations test whether the restored or recomputed blocks are correct.

A separate *reference* call uses a different UUID (guaranteeing a full recompute) and is run once per scenario as ground truth. The model itself acts as judge on each iteration, also UUID-prefixed to bust its own cache.

```
┌─────────────────────────────────────────────────────────────┐
│  Reference call (unique UUID)  →  answer  (clean recompute) │
│                                                             │
│  Target iter 1  (stable UUID)  →  answer  (cold start)      │
│  Target iter 2  (stable UUID)  →  answer  (from cache)      │
│  Target iter N  (stable UUID)  →  answer  (from cache)      │
│                                                             │
│  Judge call (unique UUID)  →  score + pass/fail per iter    │
└─────────────────────────────────────────────────────────────┘
```

A failure that appears on iteration 2+ but not iteration 1 is strong evidence of a KV cache recomputation bug rather than a model or prompt issue.

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
| `early_doc_recall` | multi-turn | Recall the **first** document loaded — tests recomputation of blocks near the attention sinks |
| `middle_doc_recall` | multi-turn | Recall the **third** document loaded — tests recomputation of mid-sequence blocks |
| `specific_fact_retrieval` | multi-turn | Retrieve a specific named entity from an early document — catches value-tensor corruption (correct topic, wrong fact) |
| `sequential_summarize` | sequential pairs | Independent summarize requests in the same chat — tests cross-turn cache contamination |

With stochastic eviction, all blocks have equal probability of being evicted. Early-document scenarios are still valuable because the first tokens of the sequence act as "attention sinks" — they absorb a disproportionate share of attention across all layers — so recomputation errors there have outsized impact on output quality.

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
