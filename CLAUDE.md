# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`kvcache-sanity-check` validates that an LLM inference server using offloaded KV cache (specifically LMCache) produces **correct outputs**, not just fast ones. Standard benchmarks only measure TTFT and throughput; this tool verifies the model actually answers correctly when KV blocks are partially evicted or recomputed during segmented prefill.

## Setup

```bash
pip install -e .
```

## Running the Check

```bash
# Basic run against a local vLLM/LMCache server
kvcache-check --target-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct

# More iterations to catch flaky failures
kvcache-check --target-url http://localhost:8000 --model <model> --iterations 5

# Use a separate, more capable model as judge
kvcache-check \
  --target-url http://localhost:8000 --model <model> \
  --judge-url http://judge:8001 --judge-model gpt-4o-mini --judge-api-key sk-...

# Verbose: print both answers even on PASS
kvcache-check --target-url http://localhost:8000 --model <model> -v
```

Exits with code 1 if any scenario fails.

## Architecture

```
kvcache_sanity/
  main.py        CLI entry point (Click)
  models.py      Pydantic data models: Document, Scenario, EvaluationResult, TestResult
  corpus.py      Load .txt documents from data/corpus/, merged with the user corpus dir
  scenarios.py   Load scenario YAML from data/scenarios/, merged with the user scenarios dir
  paths.py       user_data_dir() — XDG-based location for user corpus/scenarios
  runner.py      Build multi-turn conversations and call the inference server
  evaluator.py   LLM-as-judge comparison between target and reference answers
  report.py      Rich-formatted terminal output

  data/
    scenarios/
      default.yaml   Built-in test scenarios
      presets/       Context-length presets (4k/8k/32k/128k.yaml)
    corpus/
      doc_s1_python.txt          ~500 words — Python language history
      doc_s2_coffee.txt          ~600 words — Coffee origins
      doc_m1_apollo.txt          ~1100 words — Apollo space program
      doc_m2_dna.txt             ~1000 words — DNA double helix discovery
      doc_l1_french_revolution.txt  ~1600 words — French Revolution
```

`corpus.py`/`scenarios.py` resolve the bundled `data/` via `importlib.resources`, not `Path(__file__)` — this is what makes the corpus and default scenarios ship inside the installed wheel and still resolve correctly outside a source checkout.

**Bundled vs. user data** — with no explicit `--corpus-dir`/`--scenarios-file`/`--scenarios`, `load_documents()`/`load_scenarios()` load the bundled `data/` and merge in `user_data_dir()/corpus/*.txt` and `user_data_dir()/scenarios/*.yaml` on top (user `doc_id`/scenario `id` wins on a match). `user_data_dir()` defaults to `~/.local/share/kvcache-sanity-check` (or `$XDG_DATA_HOME/kvcache-sanity-check`), overridable via `KVCACHE_DATA_DIR`. Passing `--corpus-dir`/`--scenarios-file` explicitly bypasses this merge entirely — that directory/file is used exclusively. Note that `--corpus-dir` always governs corpus lookup regardless of which scenario source is picked, so an explicit `--corpus-dir` combined with `--scenarios NAME` for a bundled preset can fail to find documents that only exist in the bundled/user corpus.

**`--scenarios NAME`** — `scenarios.find_named_scenarios()` resolves a bare name (e.g. `32k`) against the user scenarios dir, then the bundled presets dir, then the bundled top-level `data/scenarios/` (so `--scenarios default` works too), user data winning on a name collision. This exists because `--scenarios-file` needs a literal filesystem path, and the bundled presets don't sit at a predictable relative path once installed — see `kvcache_sanity/data/scenarios/presets/`. Mutually exclusive with `--scenarios-file`.

### Evaluation design

1. **Target call** — run the scenario normally; the server may serve from cached KV blocks.
2. **Reference call** — run the same scenario with a UUID injected at the start of the system message (`[<uuid>] You are a helpful assistant…`). Because KV caches are keyed on the full token prefix, any difference in the first token guarantees a complete cache miss and a full recompute.
3. **Judge call** — ask the LLM (same server, another UUID prefix) to compare the two answers and return a JSON score (0–10) and verdict. Pass threshold defaults to 0.7.

Hardcoded assistant acknowledgments are used between document turns so the conversation structure is byte-for-byte identical between target and reference runs — the only variable is the system message prefix.

### Scenario design

Questions are intentionally about **early documents** in the conversation (loaded first → KV blocks computed first → highest eviction risk). Documents are varied enough in topic that accidentally answering about the wrong one is unambiguous.

## Extending

**Add documents without touching the install** — drop `.txt` files in `~/.local/share/kvcache-sanity-check/corpus/` (or wherever `KVCACHE_DATA_DIR` points); merged on top of the bundled corpus. First line must be `# Title`, remainder is the body. To modify the bundled set directly, edit `kvcache_sanity/data/corpus/`.

**Add scenarios without touching the install** — drop `.yaml` files in `~/.local/share/kvcache-sanity-check/scenarios/`; merged on top of `kvcache_sanity/data/scenarios/default.yaml`. Or pass `--scenarios-file path/to/custom.yaml` to use only that file.

**Custom corpus** — pass `--corpus-dir` to use only that directory (bypasses both bundled and user data).

**TODO: LMCache fault injection** — integrate with the LMCache control API to programmatically vary the KV eviction/failure rate and sweep quality-vs-eviction-rate curves. This would let the tool drive the failure rate rather than just observe it, enabling systematic characterization of the degradation boundary.
