# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-11-19

Changes since v0.1.3.

### Added

- `--accuracy-csv PATH` writes a CloudAI-compatible accuracy summary:
  `Task,Correct,Total,Accuracy,Invalid,ReferenceIncomplete,TargetIncomplete,JudgeError,TargetLoop`,
  one row per scenario plus an `OVERALL` row. Written at the end of the run regardless of the exit code.
- Judge response is constrained to a JSON schema (`response_format`, served with guided decoding
  such as xgrammar on vLLM), so it can no longer come back as prose or with an out-of-range score.
  `--no-judge-structured` restores prompt-only JSON for servers that reject `response_format`.
- `--judge-max-tokens` and `--judge-retries`: the judge is retried (fresh cache-busting prefix each
  time) when a response carries no valid verdict. A valid verdict, including FAIL, is never retried.
- `--reference-retries`: truncated or empty references are regenerated under a fresh prefix.
- Result statuses `JUDGE_ERROR`, `REFERENCE_INCOMPLETE` and `TARGET_INCOMPLETE`, shown in the report,
  the CSV and `--log-file` entries.
- Truncated generations are classified as `loop` (the tail repeats itself) or `budget`; the
  `TargetLoop` CSV column counts target answers cut off while looping next to a complete reference,
  a possible sign of corrupted KV blocks rather than a too small token budget.
- `--log-file` entries now include `status`, `reference_attempts`, `judge_attempt_details` and
  target finish/truncation details. `status` is optional in the log model, so older logs should still load.

### Changed

- **Exit code:** a judge response with no valid verdict, a truncated or empty target answer and an
  incomplete reference are all non-passing, so they make `kvcache-check` exit 1.
- **Judge:** the default judge `max_tokens` is 4096 (was a hardcoded 1024). The `reasoning` field now
  precedes `score` in the prompts, and the ignored `verdict` field was dropped. Scores may differ
  slightly from 0.1.3.
- **Scenarios:** `early_doc_recall` and the Document 3 recall scenario in `default.yaml` now ask two
  single-answer facts instead of a free-form summary, because the open request made reasoning models
  loop until `max_tokens`. Expected-answer keywords changed accordingly, so results are not
  comparable with 0.1.3 runs.
- `TestResult.status` is a required field of the Python model.

### Internal

- Run settings are shared through `RunContext`; reference preparation and judge settings
  (`JudgeConfig`) were extracted into their own units.
