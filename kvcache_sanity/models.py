from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel
from typing import Optional


class ResultStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
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
    complete: bool
    reasoning_tokens: Optional[int] = None
    truncation: Optional[str] = None
    truncated_in_reasoning: Optional[bool] = None


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
