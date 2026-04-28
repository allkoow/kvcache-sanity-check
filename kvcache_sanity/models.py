from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import Optional


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
    error: Optional[str] = None
    target_request_id: str = ""


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


# ---------------------------------------------------------------------------
# Evaluator return type — wraps EvaluationResult with the judge trace.
# ---------------------------------------------------------------------------

@dataclass
class EvaluationTrace:
    result: EvaluationResult
    judge_messages: list[dict] = field(default_factory=list)
    judge_raw_response: str = ""
    eval_prefix: str = ""


# ---------------------------------------------------------------------------
# Log record — one JSON line per run in the .jsonl log file.
# ---------------------------------------------------------------------------

class MessageLog(BaseModel):
    role: str
    content: str  # document content is truncated at log-write time


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
