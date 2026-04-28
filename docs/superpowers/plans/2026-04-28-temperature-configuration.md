# Temperature Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the inference temperature configurable via a global `--temperature` CLI flag and an optional per-scenario `temperature` field in YAML, with separate `--judge-temperature` for the evaluator, and `replay.py` reading temperature from the log entry (falling back to `--temperature` with a warning if absent).

**Architecture:** Temperature is added as an optional `float | None` field to `Scenario` (None = inherit global), threaded through `_call_api` / `run_scenario` / `get_reference_answer` / `run_sequential_pair` / `get_reference_pair_answer`, stored in `RunLog` so `replay.py` can recover it, and exposed in `main.py` as `--temperature` (default 0.1) and `--judge-temperature` (default 0.0).

**Tech Stack:** Python, Click, Pydantic, OpenAI SDK, PyYAML

---

## File Map

| File | Change |
|------|--------|
| `kvcache_sanity/models.py` | Add `temperature: float \| None = None` to `Scenario`; add `temperature: float \| None = None` to `RunLog` |
| `kvcache_sanity/runner.py` | Add `temperature: float` param to `_call_api`, `run_scenario`, `get_reference_answer`, `run_sequential_pair`, `get_reference_pair_answer` |
| `kvcache_sanity/evaluator.py` | Add `judge_temperature: float = 0.0` param to `evaluate_answers` |
| `kvcache_sanity/logger.py` | Add `temperature: float` param to `RunLogger.log`; pass it into `RunLog` |
| `kvcache_sanity/main.py` | Add `--temperature` (default 0.1) and `--judge-temperature` (default 0.0) CLI options; resolve effective temperature per scenario; thread through all calls; pass to logger |
| `kvcache_sanity/replay.py` | Add `--temperature` option (default 0.1); read `log_entry.temperature`, warn if absent, fall back to flag |

---

## Task 1 — Add `temperature` fields to models

**Files:**
- Modify: `kvcache_sanity/models.py`

- [ ] **Step 1: Add `temperature` to `Scenario`**

  In `models.py`, add the field to `Scenario` (after `evaluate_from_pair`):

  ```python
  class Scenario(BaseModel):
      id: str
      description: str
      # --- multi_turn_recall mode ---
      document_ids: list[str] = []
      question: str = ""
      target_doc_id: str = ""
      key_facts: list[str] = []
      # --- sequential_pairs mode ---
      mode: str = "multi_turn_recall"
      pairs: list[ScenarioPair] = []
      evaluate_from_pair: int = 1
      # --- per-scenario temperature override ---
      temperature: float | None = None  # None = inherit global CLI value
  ```

- [ ] **Step 2: Add `temperature` to `RunLog`**

  In `models.py`, add the field to `RunLog` (after `target_request_time`):

  ```python
  class RunLog(BaseModel):
      timestamp: str
      scenario_id: str
      iteration: int
      question: str
      target_messages: list[MessageLog]
      reference_prefix: str
      target_answer: str
      reference_answer: str
      target_request_id: str = ""
      reference_request_id: str = ""
      target_request_time: str = ""
      temperature: float | None = None
      judge_messages: list[MessageLog]
      judge_raw_response: str
      evaluation: EvaluationResult
      error: Optional[str] = None
  ```

- [ ] **Step 3: Verify the file parses**

  Run: `cd /Users/iyanok/src/kvcache-sanity-check && python -c "from kvcache_sanity.models import Scenario, RunLog; print('OK')"`

  Expected output: `OK`

- [ ] **Step 4: Commit**

  ```bash
  git add kvcache_sanity/models.py
  git commit -m "feat: add temperature field to Scenario and RunLog models"
  ```

---

## Task 2 — Thread temperature through runner.py

**Files:**
- Modify: `kvcache_sanity/runner.py`

*Depends on: Task 1*

- [ ] **Step 1: Add `temperature` param to `_call_api`**

  Replace the current signature and the hardcoded `temperature=0.1` inside `client.chat.completions.create`:

  ```python
  def _call_api(
      client: OpenAI,
      model: str,
      messages: list[dict],
      max_tokens: int,
      label: str = "",
      temperature: float = 0.1,
  ) -> tuple[str, str, int, str, str]:
      """Call the chat completions API, retrying on empty answers and warning on truncation.

      Returns (answer, finish_reason, completion_tokens, request_id, request_time).
      request_time is an ISO-8601 UTC timestamp captured immediately after the response
      is received — useful for correlating with server-side logs.
      """
      tag = f" [{label}]" if label else ""
      answer = ""
      finish_reason = ""
      completion_tokens = 0
      request_id = ""
      request_time = ""

      for attempt in range(_MAX_EMPTY_RETRIES + 1):
          response = client.chat.completions.create(
              model=model,
              messages=messages,
              max_tokens=max_tokens,
              temperature=temperature,
          )
          # ... rest of function unchanged
  ```

- [ ] **Step 2: Add `temperature` param to `run_scenario`**

  Update signature and the `_call_api` call:

  ```python
  def run_scenario(
      scenario: Scenario,
      documents: dict[str, Document],
      client: OpenAI,
      model: str,
      unique_prefix: str | None = None,
      max_tokens: int = 1024,
      temperature: float = 0.1,
  ) -> RunResult:
      messages = build_messages(scenario, documents, unique_prefix)
      label = f"scenario={scenario.id}, prefix={'ref' if unique_prefix else 'target'}"
      answer, finish_reason, completion_tokens, request_id, request_time = _call_api(
          client, model, messages, max_tokens, label, temperature=temperature
      )
      return RunResult(
          answer=answer,
          messages=messages,
          unique_prefix=unique_prefix,
          finish_reason=finish_reason,
          completion_tokens=completion_tokens,
          request_id=request_id,
          request_time=request_time,
      )
  ```

- [ ] **Step 3: Add `temperature` param to `get_reference_answer`**

  ```python
  def get_reference_answer(
      scenario: Scenario,
      documents: dict[str, Document],
      client: OpenAI,
      model: str,
      max_tokens: int = 1024,
      prefix: str | None = None,
      temperature: float = 0.1,
  ) -> RunResult:
      """Run the scenario with a cache-busting prefix to force a full KV cache miss.

      If prefix is supplied it is reused (so subsequent target calls with the same
      prefix will hit blocks cached by this reference call). If omitted a fresh UUID
      is generated, which is appropriate when the caller wants an isolated ground truth.
      """
      if prefix is None:
          prefix = str(uuid.uuid4())
      return run_scenario(scenario, documents, client, model, prefix, max_tokens, temperature=temperature)
  ```

- [ ] **Step 4: Add `temperature` param to `run_sequential_pair`**

  ```python
  def run_sequential_pair(
      pairs: list[ScenarioPair],
      documents: dict[str, Document],
      client: OpenAI,
      model: str,
      pair_index: int,
      unique_prefix: str | None = None,
      max_tokens: int = 1024,
      temperature: float = 0.1,
  ) -> RunResult:
      """Run the conversation up to pair_index and return the answer for that pair."""
      messages = build_pairs_messages(pairs, documents, pair_index, unique_prefix)
      label = f"pair={pair_index}, prefix={'ref' if unique_prefix else 'target'}"
      answer, finish_reason, completion_tokens, request_id, request_time = _call_api(
          client, model, messages, max_tokens, label, temperature=temperature
      )
      return RunResult(
          answer=answer,
          messages=messages,
          unique_prefix=unique_prefix,
          finish_reason=finish_reason,
          completion_tokens=completion_tokens,
          request_id=request_id,
          request_time=request_time,
      )
  ```

- [ ] **Step 5: Add `temperature` param to `get_reference_pair_answer`**

  ```python
  def get_reference_pair_answer(
      pairs: list[ScenarioPair],
      documents: dict[str, Document],
      client: OpenAI,
      model: str,
      pair_index: int,
      max_tokens: int = 1024,
      prefix: str | None = None,
      temperature: float = 0.1,
  ) -> RunResult:
      """Same as run_sequential_pair but with a cache-busting prefix to force a cache miss.

      If prefix is supplied it is reused so subsequent target calls with the same
      prefix will hit blocks cached by this reference call.
      """
      if prefix is None:
          prefix = str(uuid.uuid4())
      return run_sequential_pair(pairs, documents, client, model, pair_index, prefix, max_tokens, temperature=temperature)
  ```

- [ ] **Step 6: Verify the module imports cleanly**

  Run: `cd /Users/iyanok/src/kvcache-sanity-check && python -c "from kvcache_sanity.runner import _call_api, run_scenario, get_reference_answer, run_sequential_pair, get_reference_pair_answer; print('OK')"`

  Expected output: `OK`

- [ ] **Step 7: Commit**

  ```bash
  git add kvcache_sanity/runner.py
  git commit -m "feat: add temperature parameter to runner API calls"
  ```

---

## Task 3 — Add `judge_temperature` param to evaluator.py

**Files:**
- Modify: `kvcache_sanity/evaluator.py`

*Can run in parallel with Tasks 2 and 4 (after Task 1)*

- [ ] **Step 1: Update `evaluate_answers` signature**

  Add `judge_temperature: float = 0.0` parameter. The judge should default to 0.0 (deterministic) separate from the scenario temperature. Update the `eval_client.chat.completions.create` call:

  ```python
  def evaluate_answers(
      question: str,
      target_answer: str,
      reference_answer: str,
      client: OpenAI,
      model: str,
      threshold: float = 0.7,
      judge_client: OpenAI | None = None,
      judge_prompt: str = DEFAULT_PROMPT,
      judge_temperature: float = 0.0,
  ) -> EvaluationTrace:
      """Compare target_answer to reference_answer using LLM-as-judge.

      judge_prompt selects which prompt template to use: "strict" (default,
      fine-grained consistency scoring) or "topic" (only catches wrong-document
      answers, ignores detail differences).

      The evaluation call itself uses a unique prefix to prevent cache hits,
      since the judge must not be influenced by prior cached state either.

      Returns an EvaluationTrace that bundles the EvaluationResult with the full
      judge message exchange and raw response for logging.
      """
      if judge_prompt not in _PROMPTS:
          raise ValueError(f"Unknown judge_prompt {judge_prompt!r}. Choose from: {PROMPT_NAMES}")

      eval_client = judge_client or client
      eval_prefix = str(uuid.uuid4())

      messages = [
          {
              "role": "system",
              "content": (
                  f"[eval-session:{eval_prefix}] "
                  "You are a precise answer evaluator. Respond only with valid JSON."
              ),
          },
          {
              "role": "user",
              "content": _PROMPTS[judge_prompt].format(
                  question=question,
                  reference_answer=reference_answer,
                  target_answer=target_answer,
              ),
          },
      ]

      response = eval_client.chat.completions.create(
          model=model,
          messages=messages,
          max_tokens=1024,
          temperature=judge_temperature,
      )
      # ... rest of function unchanged
  ```

- [ ] **Step 2: Verify the module imports cleanly**

  Run: `cd /Users/iyanok/src/kvcache-sanity-check && python -c "from kvcache_sanity.evaluator import evaluate_answers; print('OK')"`

  Expected output: `OK`

- [ ] **Step 3: Commit**

  ```bash
  git add kvcache_sanity/evaluator.py
  git commit -m "feat: add judge_temperature parameter to evaluate_answers"
  ```

---

## Task 4 — Store temperature in RunLogger

**Files:**
- Modify: `kvcache_sanity/logger.py`

*Can run in parallel with Tasks 2 and 3 (after Task 1)*

- [ ] **Step 1: Add `temperature` param to `RunLogger.log`**

  Update the `log` method to accept and persist temperature:

  ```python
  def log(
      self,
      scenario: Scenario,
      iteration: int,
      target: RunResult,
      reference: RunResult,
      trace: EvaluationTrace,
      error: str | None = None,
      temperature: float | None = None,
  ) -> None:
      run_log = RunLog(
          timestamp=datetime.now(timezone.utc).isoformat(),
          scenario_id=scenario.id,
          iteration=iteration,
          question=scenario.question,
          target_messages=_truncate_messages(target.messages),
          reference_prefix=reference.unique_prefix or "",
          target_answer=target.answer,
          reference_answer=reference.answer,
          target_request_id=target.request_id,
          reference_request_id=reference.request_id,
          target_request_time=target.request_time,
          temperature=temperature,
          judge_messages=_truncate_messages(trace.judge_messages),
          judge_raw_response=trace.judge_raw_response,
          evaluation=trace.result,
          error=error,
      )
      with open(self._path, "a", encoding="utf-8") as f:
          f.write(run_log.model_dump_json() + "\n")
  ```

- [ ] **Step 2: Verify the module imports cleanly**

  Run: `cd /Users/iyanok/src/kvcache-sanity-check && python -c "from kvcache_sanity.logger import RunLogger; print('OK')"`

  Expected output: `OK`

- [ ] **Step 3: Commit**

  ```bash
  git add kvcache_sanity/logger.py
  git commit -m "feat: store temperature in RunLog via RunLogger"
  ```

---

## Task 5 — Wire temperature through main.py

**Files:**
- Modify: `kvcache_sanity/main.py`

*Depends on: Tasks 2, 3, 4*

- [ ] **Step 1: Add CLI options**

  Add two new Click options after `--judge-api-key`:

  ```python
  @click.option("--temperature", default=0.1, show_default=True, type=float,
                help="Sampling temperature for scenario runs (0.0 = deterministic). "
                     "Per-scenario YAML 'temperature' field overrides this.")
  @click.option("--judge-temperature", default=0.0, show_default=True, type=float,
                help="Sampling temperature for the LLM-as-judge evaluation call.")
  ```

  Add both to the `cli` function signature:

  ```python
  def cli(
      target_url, model, api_key,
      judge_url, judge_model, judge_api_key,
      temperature, judge_temperature,
      iterations, threshold, scenarios_file, corpus_dir,
      max_tokens, judge_prompt, log_file, verbose,
  ):
  ```

- [ ] **Step 2: Print temperature in the startup banner**

  After the existing `console.print(f"Scenarios: ...")` line:

  ```python
  console.print(f"Temperature: {temperature}  Judge temperature: {judge_temperature}")
  ```

- [ ] **Step 3: Add `temperature` and `judge_temperature` to `common` dict**

  In the per-scenario loop, compute the effective temperature and pass it through. Replace the `common = dict(...)` block:

  ```python
  for scenario in scenarios:
      console.print(f"[bold cyan]{scenario.id}[/] — {scenario.description}")

      effective_temp = scenario.temperature if scenario.temperature is not None else temperature

      common = dict(
          scenario=scenario, documents=documents,
          target_client=target_client, model=model,
          judge_model_name=judge_model_name, judge_client=judge_client,
          threshold=threshold, max_tokens=max_tokens, judge_prompt=judge_prompt,
          iterations=iterations, all_results=all_results, run_logger=run_logger, verbose=verbose,
          temperature=effective_temp, judge_temperature=judge_temperature,
      )
  ```

- [ ] **Step 4: Update `_run_scenario_iterations` to accept and use temperature**

  Update the function signature and all runner/evaluator/logger calls:

  ```python
  def _run_scenario_iterations(
      *, scenario, documents, target_client, model, judge_model_name, judge_client,
      threshold, max_tokens, judge_prompt, iterations, all_results, run_logger, verbose,
      temperature, judge_temperature,
  ) -> None:
      shared_prefix = str(uuid.uuid4())

      reference_run = RunResult(answer="")
      ref_error: str | None = None
      try:
          reference_run = get_reference_answer(
              scenario, documents, target_client, model,
              max_tokens=max_tokens, prefix=shared_prefix,
              temperature=temperature,
          )
      except Exception as exc:
          ref_error = str(exc)

      for i in range(1, iterations + 1):
          error = ref_error
          target_run = RunResult(answer="")
          trace = EvaluationTrace(result=EvaluationResult(score=0.0, reasoning="", passed=False))

          if ref_error:
              result = TestResult(
                  scenario_id=scenario.id, iteration=i, question=scenario.question,
                  target_answer="", reference_answer="",
                  evaluation=EvaluationResult(score=0.0, reasoning=ref_error, passed=False),
                  error=ref_error,
              )
          else:
              try:
                  target_run = run_scenario(
                      scenario, documents, target_client, model,
                      unique_prefix=shared_prefix, max_tokens=max_tokens,
                      temperature=temperature,
                  )
                  trace = evaluate_answers(
                      question=scenario.question,
                      target_answer=target_run.answer,
                      reference_answer=reference_run.answer,
                      client=target_client,
                      model=judge_model_name,
                      threshold=threshold,
                      judge_client=judge_client,
                      judge_prompt=judge_prompt,
                      judge_temperature=judge_temperature,
                  )
                  result = TestResult(
                      scenario_id=scenario.id, iteration=i, question=scenario.question,
                      target_answer=target_run.answer, reference_answer=reference_run.answer,
                      evaluation=trace.result,
                      target_request_id=target_run.request_id,
                  )
              except Exception as exc:
                  error = str(exc)
                  result = TestResult(
                      scenario_id=scenario.id, iteration=i, question=scenario.question,
                      target_answer=target_run.answer, reference_answer=reference_run.answer,
                      evaluation=EvaluationResult(score=0.0, reasoning=str(exc), passed=False),
                      error=error,
                  )

          all_results.append(result)
          report.print_result(result, verbose=verbose)
          if run_logger:
              run_logger.log(scenario=scenario, iteration=i,
                             target=target_run, reference=reference_run, trace=trace, error=error,
                             temperature=temperature)
  ```

- [ ] **Step 5: Update `_run_sequential_pairs` to accept and use temperature**

  Update the function signature and all runner/evaluator/logger calls identically:

  ```python
  def _run_sequential_pairs(
      *, scenario, documents, target_client, model, judge_model_name, judge_client,
      threshold, max_tokens, judge_prompt, iterations, all_results, run_logger, verbose,
      temperature, judge_temperature,
  ) -> None:
      pairs = scenario.pairs
      shared_prefix = str(uuid.uuid4())

      for pair_idx in range(scenario.evaluate_from_pair, len(pairs)):
          pair = pairs[pair_idx]
          pair_label = f"pair {pair_idx + 1}/{len(pairs)} (doc={pair.doc_id})"

          reference_run = RunResult(answer="")
          ref_error: str | None = None
          try:
              reference_run = get_reference_pair_answer(
                  pairs, documents, target_client, model, pair_idx,
                  max_tokens=max_tokens, prefix=shared_prefix,
                  temperature=temperature,
              )
          except Exception as exc:
              ref_error = str(exc)

          for i in range(1, iterations + 1):
              error = ref_error
              target_run = RunResult(answer="")
              trace = EvaluationTrace(result=EvaluationResult(score=0.0, reasoning="", passed=False))

              if ref_error:
                  result = TestResult(
                      scenario_id=f"{scenario.id}[{pair_label}]",
                      iteration=i, question=pair.question,
                      target_answer="", reference_answer="",
                      evaluation=EvaluationResult(score=0.0, reasoning=ref_error, passed=False),
                      error=ref_error,
                  )
              else:
                  try:
                      target_run = run_sequential_pair(
                          pairs, documents, target_client, model, pair_idx,
                          unique_prefix=shared_prefix, max_tokens=max_tokens,
                          temperature=temperature,
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
                          judge_temperature=judge_temperature,
                      )
                      result = TestResult(
                          scenario_id=f"{scenario.id}[{pair_label}]",
                          iteration=i, question=pair.question,
                          target_answer=target_run.answer, reference_answer=reference_run.answer,
                          evaluation=trace.result,
                          target_request_id=target_run.request_id,
                      )
                  except Exception as exc:
                      error = str(exc)
                      result = TestResult(
                          scenario_id=f"{scenario.id}[{pair_label}]",
                          iteration=i, question=pair.question,
                          target_answer=target_run.answer, reference_answer=reference_run.answer,
                          evaluation=EvaluationResult(score=0.0, reasoning=str(exc), passed=False),
                          error=error,
                      )

              all_results.append(result)
              report.print_result(result, verbose=verbose)
              if run_logger:
                  run_logger.log(scenario=scenario, iteration=i,
                                 target=target_run, reference=reference_run, trace=trace, error=error,
                                 temperature=temperature)
  ```

- [ ] **Step 6: Verify the CLI help renders without error**

  Run: `cd /Users/iyanok/src/kvcache-sanity-check && python -m kvcache_sanity.main --help`

  Expected: help text printed with `--temperature` and `--judge-temperature` options visible, no traceback.

- [ ] **Step 7: Commit**

  ```bash
  git add kvcache_sanity/main.py
  git commit -m "feat: add --temperature and --judge-temperature CLI options"
  ```

---

## Task 6 — Add temperature support to replay.py

**Files:**
- Modify: `kvcache_sanity/replay.py`

*Can run in parallel with Task 5 (after Tasks 2 and 4)*

- [ ] **Step 1: Add `--temperature` CLI option**

  Add the option to the `@click.command()` decorator block (after `--max-tokens`):

  ```python
  @click.option("--temperature", default=0.1, show_default=True, type=float,
                help="Sampling temperature used if the log entry does not record one.")
  ```

  Add `temperature` to the function signature:

  ```python
  def replay(
      log_file, request_id, count,
      target_url, model, api_key, max_tokens, temperature,
      scenarios_file, corpus_dir,
  ) -> None:
  ```

- [ ] **Step 2: Resolve temperature from log entry with fallback warning**

  After finding `entry` and before the `console.print(f"Found: ...")` block, add:

  ```python
  if entry.temperature is not None:
      effective_temperature = entry.temperature
  else:
      console.print(
          f"[yellow]WARNING[/yellow]: log entry has no temperature recorded; "
          f"using --temperature={temperature}"
      )
      effective_temperature = temperature
  ```

  Then print the resolved temperature in the banner:

  ```python
  console.print(f"Found: [bold]{entry.scenario_id}[/]  iter {entry.iteration}  "
                f"original @ {entry.target_request_time or entry.timestamp}")
  console.print(f"UUID prefix: {entry.reference_prefix}")
  console.print(f"Temperature: {effective_temperature}")
  console.print()
  ```

- [ ] **Step 3: Pass `temperature` to `_call_api`**

  Update the `_call_api` call in the replay loop:

  ```python
  for i in range(1, count + 1):
      answer, finish_reason, completion_tokens, req_id, req_time = _call_api(
          client, model, messages, max_tokens,
          label=f"replay {i}/{count}",
          temperature=effective_temperature,
      )
  ```

- [ ] **Step 4: Verify the CLI help renders without error**

  Run: `cd /Users/iyanok/src/kvcache-sanity-check && python -m kvcache_sanity.replay --help`

  Expected: help text printed with `--temperature` option visible, no traceback.

- [ ] **Step 5: Commit**

  ```bash
  git add kvcache_sanity/replay.py
  git commit -m "feat: replay reads temperature from log, falls back to --temperature flag"
  ```

---

## Execution Order

```
Task 1 (models.py)
    ├── Task 2 (runner.py)   ─┐
    ├── Task 3 (evaluator.py) ├─ parallel
    └── Task 4 (logger.py)   ─┘
                                  ├── Task 5 (main.py)   ─┐ parallel
                                  └── Task 6 (replay.py) ─┘
```
