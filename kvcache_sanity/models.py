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
