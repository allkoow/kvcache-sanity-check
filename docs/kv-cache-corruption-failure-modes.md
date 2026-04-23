# KV Cache Corruption: Failure Modes and Expected Behavior

## Background

This document summarizes research into how incorrect KV cache blocks manifest in LLM
output, and whether the failure modes targeted by `kvcache-sanity-check` are realistic
signals of a genuine inference bug.

## What LMCache Actually Does

When a KV block is evicted (offloaded to CPU/disk), LMCache does **not** simply discard
it and let the model attend to garbage. On a subsequent request that needs that block,
LMCache either:

1. Restores the offloaded block from CPU/disk back to GPU, or
2. Re-does the prefill computation for those tokens from scratch.

The correctness requirement is that the restored or recomputed block must be numerically
identical (or close enough) to what would have been produced if the block had stayed on
the GPU the entire time. **This tool tests whether that requirement holds.**

Eviction in the tested configuration is stochastic — any block in the sequence has
roughly equal probability of being evicted, not just early ones.

## The Core Mechanism

KV cache blocks store the attention key (K) and value (V) tensors for previously computed
tokens. If a block is recomputed incorrectly — due to a bug in the recomputation path,
wrong token indexing at block boundaries, incorrect positional encodings, or a race
condition — those K/V tensors no longer represent the actual token content.

When the model attends over a mix of correct and incorrectly recomputed blocks, the
softmax normalization acts as a stabilizer. Attention weights remain bounded
probabilities that sum to 1.0; the mechanism does not crash. Instead it produces
coherent output that is factually wrong, because the attended representations no longer
faithfully encode the original tokens.

## Failure Modes

### 1. Subtle Wrong-Document Answers (silent correctness failure)

The most dangerous mode because it is invisible without an independent correctness check.
The model produces fluent, confident output that answers the question from the wrong
document or with wrong facts.

With recomputation bugs specifically, this happens when the incorrectly recomputed K/V
values for document A happen to resemble those of a different document B (e.g., due to
token boundary misalignment or off-by-one errors in block indexing). The model attends
to the right token *positions* but with representations that encode the wrong content.

A related trigger documented in the literature is **partial recomputation**: when only
some blocks for a document are recomputed while others remain stale. Semantic entities
that span multiple tokens can be split across updated and stale blocks, producing blended
answers from two documents. The CacheClip paper (arXiv:2510.10129) demonstrates this in
RAG scenarios: an entity tokenized across a block boundary is recovered correctly on one
side and stale on the other, producing a coherent but factually wrong answer.

### 2. Garbled or Repetitive Output

More visible corruption that occurs when recomputed blocks are substantially wrong across
many layers simultaneously. The model produces structurally broken output: numbered lists
that restart mid-sequence, repeated phrases, oscillation between two topics, or
degenerate patterns like `1000…1000…1000…` until the token limit.

The KV cache manipulation attack paper (arXiv:2511.12752) characterizes this: partial
corruption across layers causes oscillating answers, while complete corruption causes
immediate incoherence.

### 3. Empty or Truncated Output

Occurs with lower-level bugs in the block allocator or recomputation pipeline. When
generation receives a zeroed or uninitialized KV block, the model emits token ID 0 for
every output position, producing empty or padding content with `finish_reason=stop` and
`completion_tokens=0`. Also observed as truncated output when `finish_reason=length`
fires earlier than expected due to malformed block state.

This is the most operationally visible failure and has been seen in practice in this
testing setup (see retry logic in `runner.py`).

## Attention Sinks: Why Any Block Position Matters

Even with uniform eviction probability, some positions have outsized impact when
recomputed incorrectly. Research (StreamingLLM, arXiv:2309.17453) shows that models
learn to concentrate 45–55% of total attention mass on the first 1–2 tokens of a
sequence — "attention sinks" — across nearly all heads and layers. These tokens act as
a no-op anchor that absorbs surplus attention.

If the KV blocks corresponding to those sink tokens are incorrectly recomputed, the
effect is disproportionate: a large fraction of the model's attention mass is now routed
through wrong representations, even though the corrupt blocks represent only a small
fraction of the sequence. This does not change the probability of eviction, but it means
that bugs triggered on early blocks may produce more dramatic failures than the same
bug triggered on a middle-of-sequence block.

## Asymmetry Between K and V

K and V tensors have different sensitivity to recomputation errors:

- **Keys (K)** control attention routing via the softmax dot product and require high
  precision. A bug that corrupts K changes *which* tokens the model attends to.
- **Values (V)** are weighted sums applied after attention weights are computed and
  tolerate more noise. Corruption of V degrades the *content* retrieved from attended
  tokens but does not reroute attention.

In practice: K recomputation errors tend to produce wrong-document answers (attention
rerouted to different positions); V recomputation errors tend to produce wrong-detail
answers about the correct document (correct topic, corrupted facts).

## Implications for kvcache-sanity-check

### What the tool tests

The tool compares the model's answer on a cached request (target) against its answer on
a forced-recompute request (reference, UUID-prefixed). A divergence means LMCache
produced a different result than a clean GPU-side computation — which is a correctness
bug by definition, since LMCache's contract is bit-identical output.

The wrong-document scenarios (`early_doc_recall`, `sequential_summarize`) are designed
to surface silent correctness failures. The specific-fact scenario (`specific_fact_retrieval`)
targets V-level corruption: correct topic, wrong named entity.

### The cold-start iteration

With the current design (one stable `target_prefix` per scenario shared across all
iterations, reference run once):

- **Iteration 1**: cold start — no blocks cached for this prefix. Any failure here
  reflects a server-side bug independent of caching (OOM, empty-slot use-after-free,
  malformed request handling).
- **Iteration 2+**: blocks from iteration 1 are now in LMCache's store (possibly
  offloaded to CPU/disk). These iterations test whether restored or recomputed blocks
  produce correct output.

A failure appearing only on iteration 2+ and not on iteration 1 is strong evidence of a
KV cache bug rather than a model or prompt issue.

### Distinguishing model limitations from cache bugs

If both target and reference answers are wrong (both over-summarize, both cite the wrong
document), the failure is a **model limitation** — the judge comparison passes because
both answers are equally weak.

If the target is wrong but the reference is correct, the failure is a **cache bug** —
the reference's UUID prefix forced a clean recompute that produced the right answer,
proving the model is capable of the correct answer when not served from a corrupt cache.

The `key_facts` field in scenario definitions (currently unused) is intended for a future
independent correctness check: even if target ≈ reference (both equally weak), the tool
could fail the scenario if neither answer contains the expected named entities.

## References

- vLLM issue #36311 — non-deterministic output with APC under concurrent eviction pressure
- vLLM issue #37076 — use-after-free in KV block allocator producing zeroed output
- [A First Look at Bugs in LLM Inference Engines](https://arxiv.org/abs/2506.09713) — systematic study of 1,050 bugs across vLLM, TensorRT-LLM, llama.cpp, ollama; 11 KV cache management bugs categorized
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — attention sink mechanism; disproportionate attention concentration on early tokens
- [CacheClip: Accelerating RAG with Effective KV Cache Reuse](https://arxiv.org/abs/2510.10129) — partial recomputation causing wrong-entity answers at block boundaries in RAG
- [Whose Narrative is it Anyway? A KV Cache Manipulation Attack](https://arxiv.org/abs/2511.12752) — empirical characterization of all three failure modes under controlled block corruption
- [LMCache Tech Report](https://lmcache.ai/tech_report.pdf) — correctness requirements for offloaded KV blocks; bit-identical output guarantee
