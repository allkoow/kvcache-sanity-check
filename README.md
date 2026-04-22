# kvcache-sanity-check

Most LLM benchmarks measure throughput and time-to-first-token — they send a prompt in and check that *something* came back quickly. That means a server can score well while returning garbage. This tool checks that your inference server is returning **correct** answers, not just fast ones.

The primary target is [LMCache](https://github.com/LMCache/LMCache) with segmented prefill: when some KV cache blocks are missing and must be recomputed, this tool verifies the recomputation happened correctly and the model's answers are still accurate.

## How it works

Each test scenario loads several documents into the model's context through a multi-turn conversation, then asks a question about one of the **earlier** documents — the ones whose KV blocks were computed first and are most likely to have been evicted or incorrectly recomputed.

To get a trustworthy reference answer without spinning up a second model, the same server is called again with a unique UUID injected at the start of the system message. Because KV caches are keyed on the full token prefix, a single changed token at position 0 guarantees a complete cache miss and a clean recompute. The two answers are then compared using the model itself as a judge (also UUID-prefixed, also cache-busted).

```
┌─────────────────────────────────────────────────────┐
│  Target call  →  answer A  (may use cached blocks)  │
│  Reference call (UUID prefix)  →  answer B  (clean) │
│  Judge call (UUID prefix)  →  score + pass/fail      │
└─────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e .
```

Requires Python 3.10+.

## Usage

```bash
# Minimal — point at a running vLLM/LMCache server
kvcache-check --target-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct

# Run each scenario 5 times to catch intermittent failures
kvcache-check --target-url http://localhost:8000 --model <model> --iterations 5

# Tighter pass threshold (default is 0.7)
kvcache-check --target-url http://localhost:8000 --model <model> --threshold 0.85

# Use a separate, stronger model as judge
kvcache-check \
  --target-url http://localhost:8000 --model <model> \
  --judge-url https://api.openai.com --judge-model gpt-4o-mini --judge-api-key sk-...

# Show both answers for every test, not just failures
kvcache-check --target-url http://localhost:8000 --model <model> --verbose
```

Exits with code 0 if all scenarios pass, 1 if any fail — suitable for CI.

## Default scenarios

| Scenario | What it tests |
|---|---|
| `early_doc_recall` | Summarise the **first** document loaded — highest eviction risk |
| `middle_doc_recall` | Summarise the **third** document loaded — moderate eviction risk |
| `specific_fact_retrieval` | Retrieve a named specific fact from the first document |

The bundled corpus contains five documents at varying lengths (~500–1600 words): Python language history, coffee origins, the Apollo program, the DNA double helix discovery, and the French Revolution. Topics are distinct enough that answering about the wrong one is unambiguous.

## Extending

**Add documents** — drop `.txt` files in `corpus/`. First line must be `# Title`, remainder is body text.

**Add scenarios** — append to `scenarios/default.yaml` or pass `--scenarios-file path/to/custom.yaml`. Each scenario specifies which documents to load (in order), the question to ask, and which document the answer should reference.

**Custom corpus** — pass `--corpus-dir /path/to/docs`.

## Roadmap

- [ ] LMCache control API integration — programmatically set eviction/failure rate and sweep quality vs. eviction-rate curves rather than just observing
- [ ] Embedding-based similarity as an alternative to LLM-as-judge
- [ ] Structured JSON output for integration with dashboards / CI reporters
