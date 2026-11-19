from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel
from typing import Optional


class ResultStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    # The request got no response within --request-timeout.
    TIMEOUT = "TIMEOUT"
    # No complete reference after all attempts: the case says nothing about the cache path.
    REFERENCE_INCOMPLETE = "REFERENCE_INCOMPLETE"
    # Truncated or empty target next to a complete reference: counted as an incorrect answer.
    TARGET_INCOMPLETE = "TARGET_INCOMPLETE"
    # No valid judge verdict after all attempts: the case says nothing about the cache path.
    JUDGE_ERROR = "JUDGE_ERROR"


class Document(BaseModel):
    id: str
    title: str
    content: str
    approximate_tokens: int


class ScenarioPair(BaseModel):
    """A single (document, question) pair used in sequential_pairs mode."""
    doc_id: str
    question: str


class Scenario(BaseModel):
    id: str
    description: str
    # --- multi_turn_recall mode ---
    document_ids: list[str] = []   # ordered; loaded into context in this sequence
    question: str = ""
    target_doc_id: str = ""        # which document the question is about
    key_facts: list[str] = []      # reserved for future keyword-based evaluation fallback
    # --- sequential_pairs mode ---
    mode: str = "multi_turn_recall"
    pairs: list[ScenarioPair] = []
    evaluate_from_pair: int = 1    # skip this many leading pairs before evaluating
    temperature: float | None = None


class EvaluationResult(BaseModel):
    score: float  # 0.0–1.0
    reasoning: str
    passed: bool


class TestResult(BaseModel):
    scenario_id: str
    iteration: int
    question: str
    target_answer: str
    reference_answer: str
    evaluation: EvaluationResult
    status: ResultStatus
    error: Optional[str] = None
    target_request_id: str = ""
    target_truncation: Optional[str] = None


# ---------------------------------------------------------------------------
# Runner return type — carries messages alongside the answer so the caller
# can build a RunLog without a second round-trip.
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    answer: str
    messages: list[dict] = field(default_factory=list)
    unique_prefix: Optional[str] = None
    finish_reason: str = ""
    completion_tokens: int = 0
    request_id: str = ""
    request_time: str = ""
    started_at: str = ""   # ISO-8601 UTC, just before the request was sent
    finished_at: str = ""  # ISO-8601 UTC, once the response or the error came back
    error: Optional[str] = None  # set when the request failed instead of returning a response
    timed_out: bool = False
    reasoning_tokens: Optional[int] = None  # only when the server reports it
    # Set only for finish_reason == "length": "loop" when the output tail repeats itself, else "budget".
    truncation: Optional[str] = None
    # Set only for finish_reason == "length": True when the cut happened before the final answer began.
    truncated_in_reasoning: Optional[bool] = None


# ---------------------------------------------------------------------------
# Evaluator return type — wraps EvaluationResult with the judge trace.
# ---------------------------------------------------------------------------

class JudgeAttemptLog(BaseModel):
    prefix: str
    raw_response: str
    finish_reason: str
    completion_tokens: int
    request_id: str = ""
    problem: Optional[str] = None  # why the response carried no valid verdict; None when it did


@dataclass
class EvaluationTrace:
    result: EvaluationResult
    judge_messages: list[dict] = field(default_factory=list)
    judge_raw_response: str = ""
    eval_prefix: str = ""
    judge_attempts: int = 1
    attempt_logs: list[JudgeAttemptLog] = field(default_factory=list)
    judge_error: Optional[str] = None  # set when no attempt produced a valid verdict


# ---------------------------------------------------------------------------
# Log record — one JSON line per run in the .jsonl log file.
# ---------------------------------------------------------------------------

class MessageLog(BaseModel):
    role: str
    content: str  # document content is truncated at log-write time


class ReferenceAttemptLog(BaseModel):
    prefix: str
    answer: str
    finish_reason: str
    completion_tokens: int
    request_id: str = ""
    request_time: str = ""
    started_at: str = ""
    finished_at: str = ""
    complete: bool
    reasoning_tokens: Optional[int] = None
    truncation: Optional[str] = None
    truncated_in_reasoning: Optional[bool] = None
    error: Optional[str] = None


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
    target_finish_reason: str = ""
    target_completion_tokens: int = 0
    target_reasoning_tokens: Optional[int] = None
    target_truncation: Optional[str] = None
    target_truncated_in_reasoning: Optional[bool] = None
    temperature: float | None = None
    judge_messages: list[MessageLog]
    judge_raw_response: str
    judge_attempts: int = 1
    judge_attempt_details: list[JudgeAttemptLog] = []
    evaluation: EvaluationResult
    status: ResultStatus | None = None  # absent in logs written before statuses existed
    reference_attempts: list[ReferenceAttemptLog] = []
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Phase artifacts — written to --state-dir so phases can run with checkpoints between them.
# ---------------------------------------------------------------------------

class Case(BaseModel):
    """One evaluated question: the exact conversation sent, without a cache-busting prefix."""

    case_id: str
    scenario_id: str
    question: str
    messages: list[dict]
    # Detects a corpus or scenario change between the phase that wrote an artifact and the one reading it.
    messages_sha256: str
    temperature: float | None = None  # scenario override of the run temperature


class ReferenceRecord(BaseModel):
    """Reference answer for one case: PASS when complete, else REFERENCE_INCOMPLETE, ERROR or TIMEOUT."""

    case: Case
    prefix: str  # prefix of the accepted (last) attempt; replays reuse it to read its cached blocks
    answer: str
    status: ResultStatus
    attempts: list[ReferenceAttemptLog] = []
    error: Optional[str] = None


class ReferenceSet(BaseModel):
    schema_version: int = 1
    created_at: str
    model: str
    temperature: float
    max_tokens: int
    records: list[ReferenceRecord]


class PrefixMode(str, Enum):
    """Which cache-busting prefix a replay sends each case under."""

    # The reference prefix: reads the blocks the reference phase stored.
    REUSE = "reuse"
    # A new prefix per case, shared by every iteration of the replay: the first request writes, the rest read.
    FRESH = "fresh"
    # The prefix an earlier replay used, e.g. to read after a reconfiguration what was written before it.
    FROM = "from"


class ReplayAttempt(BaseModel):
    """One replayed request: PASS when the generation is complete, else TARGET_INCOMPLETE, ERROR or TIMEOUT.

    A case without a complete reference is not sent and carries the reference status instead.
    """

    case_id: str
    seq: int
    prefix: str
    started_at: str = ""
    finished_at: str = ""
    status: ResultStatus
    answer: str = ""
    finish_reason: str = ""
    completion_tokens: int = 0
    request_id: str = ""
    truncation: Optional[str] = None
    error: Optional[str] = None


class ReplayRun(BaseModel):
    name: str
    prefix_mode: PrefixMode
    prefix_source: Optional[str] = None  # the earlier replay, set only for PrefixMode.FROM
    started_at: str
    finished_at: str
    attempts: list[ReplayAttempt]
