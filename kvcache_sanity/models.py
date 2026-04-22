from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import Optional


class Document(BaseModel):
    id: str
    title: str
    content: str
    approximate_tokens: int


class Scenario(BaseModel):
    id: str
    description: str
    document_ids: list[str]  # ordered; loaded into context in this sequence
    question: str
    target_doc_id: str  # which document the question is about (for metadata/reporting)
    # key_facts: reserved for future keyword-based evaluation fallback
    key_facts: list[str] = []


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


# ---------------------------------------------------------------------------
# Runner return type — carries messages alongside the answer so the caller
# can build a RunLog without a second round-trip.
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    answer: str
    messages: list[dict] = field(default_factory=list)
    unique_prefix: Optional[str] = None


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
    judge_messages: list[MessageLog]
    judge_raw_response: str
    evaluation: EvaluationResult
    error: Optional[str] = None
