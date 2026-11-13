# kvcache-sanity-check

Most LLM benchmarks measure throughput and time-to-first-token — they send a prompt in and check that *something* came back quickly. That means a server can score well while returning garbage. This tool checks that your inference server is returning **correct** answers, not just fast ones.

The primary target is [LMCache](https://github.com/LMCache/LMCache) with segmented prefill: when some KV cache blocks are missing and must be recomputed, this tool verifies the recomputation happened correctly and the model's answers are still accurate.

## How it works

Each test scenario loads several documents into the model's context through a multi-turn conversation, then asks a question that requires recalling a specific document. Scenarios are designed so the question targets a document whose answer is unambiguous — answering from the wrong document is obvious.

For each scenario, the tool generates one UUID shared by the reference call and all target iterations. The reference call runs **first**, giving the server a clean full recompute that both establishes the ground truth answer and warms the KV cache. Every subsequent target iteration reuses the same UUID, so it hits the blocks cached by the reference call — which LMCache may have offloaded to CPU/disk in the meantime.

```
┌──────────────────────────────────────────────────────────────────┐
│  Reference call  (UUID X)  →  answer  (clean recompute, warms cache) │
│                                                                  │
│  Target iter 1   (UUID X)  →  answer  (from cache)              │
│  Target iter 2   (UUID X)  →  answer  (from cache)              │
│  Target iter N   (UUID X)  →  answer  (from cache)              │
│                                                                  │
│  Judge call (fresh UUID)  →  score + pass/fail per iter          │
└──────────────────────────────────────────────────────────────────┘
```

Every target iteration exercises the cache equally — there is no cold-start iteration. A divergence between any target answer and the reference is a KV cache correctness bug.

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

## Context-length presets

`kvcache_sanity/data/scenarios/presets/` has one scenario file per target context window, sized to sit comfortably below that window (not right at the edge) so a run against a server configured for that window doesn't error out on a snug prompt. Each file's header comment shows the token math and the `--max-tokens` value it assumes.

| Preset | Window | Recall target | Recommended flags |
|---|---|---|---|
| `4k` | 4,096 | first (smallest) document | `--max-tokens 256` |
| `8k` | 8,192 | first document | `--max-tokens 512` |
| `32k` | 32,768 | first + middle document | `--max-tokens 1024` (default) |
| `128k` | 131,072 | first document | `--max-tokens 2048` |

```bash
kvcache-check --target-url http://localhost:8000 --model <model> \
  --scenarios 32k --max-tokens 1024
```

`--scenarios NAME` resolves a preset by name against the user scenarios dir first, then the bundled presets, then the bundled `default.yaml` (so `--scenarios default` also works) — this is what makes presets reachable from an installed package, where the bundled files no longer sit at a predictable path relative to your working directory. To point at a scenario file directly instead, use `--scenarios-file path/to/file.yaml` (see below); the two flags are mutually exclusive.

Document token counts are the corpus's rough `chars/4` estimate (see `approximate_tokens` in `corpus.py`), not an exact tokenizer count — that's why each preset leaves ~10–18% headroom rather than targeting the window exactly. If your server's actual tokenizer produces meaningfully more tokens per document than this estimate, trim a document from the list or raise `--max-tokens` down accordingly.

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

**Add scenarios/documents without touching the install** — drop `.yaml` scenario files or `.txt` corpus documents into the user data directory and they're merged on top of the bundled ones (a user `id`/`doc_id` that matches a bundled one wins):

```
~/.local/share/kvcache-sanity-check/scenarios/*.yaml
~/.local/share/kvcache-sanity-check/corpus/*.txt
```

This follows the XDG Base Directory spec (`$XDG_DATA_HOME/kvcache-sanity-check` if set), or override the whole location with `KVCACHE_DATA_DIR=/path/to/dir`.

**Full override instead of merge** — pass `--scenarios-file path/to/custom.yaml` or `--corpus-dir /path/to/docs` to use *only* that file/directory, ignoring both the bundled and user data. Documents must be `.txt` with `# Title` on the first line.

Corpus lookup itself isn't affected by which scenario file/preset you pick — `load_documents()` always does its own bundled+user merge (see above) unless you pass `--corpus-dir` explicitly, in which case *only* that directory is searched, even for a `--scenarios NAME` preset that expects a document only present in the bundled or user corpus.

## Roadmap

- [ ] LMCache control API integration — programmatically set eviction/failure rate and sweep quality vs. eviction-rate curves rather than just observing
- [ ] Embedding-based similarity as an alternative to LLM-as-judge
- [ ] Structured JSON output for integration with dashboards / CI reporters
