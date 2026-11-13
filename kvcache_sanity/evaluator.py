import json
import uuid
from dataclasses import dataclass
from openai import OpenAI
from rich.console import Console as _Console
from rich.markup import escape
from kvcache_sanity.models import EvaluationResult, EvaluationTrace, JudgeAttemptLog

_stderr = _Console(stderr=True)

# gpt-oss emits harmony channels ("analysis…assistantfinal…") inside `content` when the
# server runs without a reasoning parser; only the final channel carries the verdict.
_HARMONY_FINAL_MARKER = "assistantfinal"


def _strip_reasoning(raw: str) -> str:
    marker_pos = raw.rfind(_HARMONY_FINAL_MARKER)
    if marker_pos == -1:
        return raw
    return raw[marker_pos + len(_HARMONY_FINAL_MARKER):].strip()


def _extract_json(raw: str) -> str:
    """Best-effort extraction of a JSON object from a free-form model response.

    Handles three common cases:
    - Bare JSON (ideal)
    - JSON wrapped in ```...``` or ```json...``` code fences
    - JSON embedded in prose (find first { … last })
    """
    # 1. Bare JSON — try it directly first (fast path)
    stripped = raw.strip()
    if stripped.startswith("{"):
        return stripped

    # 2. Markdown code fence
    if "```" in raw:
        for part in raw.split("```"):
            candidate = part.lstrip("json").strip()
            if candidate.startswith("{"):
                return candidate

    # 3. JSON embedded in prose — extract from first { to last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return raw[start:end + 1]

    return raw  # give up; caller will get a parse error with the full text


_PROMPTS: dict[str, str] = {}

# strict: fine-grained consistency scoring (default)
_PROMPTS["strict"] = """\
You are evaluating whether two answers to the same question are consistent and correct.

Question asked: {question}

REFERENCE answer (generated with a clean recompute, no cached KV blocks — treat as ground truth):
{reference_answer}

TARGET answer (generated using potentially cached KV blocks — this is what you are evaluating):
{target_answer}

Evaluate whether the TARGET answer is consistent with the REFERENCE answer on:
1. Topic/document addressed — does it talk about the same thing?
2. Key facts and information — are the facts consistent?
3. Overall accuracy — would a reader get the same understanding?

Score 0–10:
  10 = identical or essentially equivalent
  7–9 = consistent, minor wording differences
  4–6 = partially correct or missing key points
  0–3 = wrong topic, major factual errors, or about a different document entirely

Respond with valid JSON only — no markdown, no explanation outside the JSON:
{{"score": <int 0-10>, "reasoning": "<one or two sentences>", "verdict": "<pass|fail>"}}\
"""

# topic: only catches wrong-document answers; ignores detail differences
_PROMPTS["topic"] = """\
You are checking whether an AI answer addresses the correct source document.

Question asked: {question}

REFERENCE answer (correct, from a clean recompute — use this to identify the expected topic):
{reference_answer}

TARGET answer (under test — may have used a corrupted KV cache):
{target_answer}

Your only job: determine whether the TARGET answer is about the SAME document or topic as the
REFERENCE answer, or whether it is clearly about a COMPLETELY DIFFERENT document or topic.

Do NOT penalise for different wording, missing details, extra information, or style.
Ignore how the source is labelled — "document", "book", "article", "text", "passage" all mean
the same thing here. ONLY score low if the TARGET is clearly speaking about a different
subject entirely (e.g. Ancient Rome when the reference is about the Internet).

First identify the topic of each answer in a short phrase. Then score:
  8–10 = same topic/document (even if details differ)
  4–7  = ambiguous — possibly the right topic but hard to tell
  0–3  = clearly about a different topic or document entirely

Respond with valid JSON only — no markdown, no explanation outside the JSON:
{{"reference_topic": "<topic of REFERENCE in a few words>", "target_topic": "<topic of TARGET in a few words>", "score": <int 0-10>, "reasoning": "<one sentence>", "verdict": "<pass|fail>"}}\
"""

PROMPT_NAMES = sorted(_PROMPTS)
DEFAULT_PROMPT = "strict"
DEFAULT_JUDGE_MAX_TOKENS = 4096
DEFAULT_JUDGE_RETRIES = 2


@dataclass(frozen=True)
class JudgeConfig:
    """How to call the LLM-as-judge and when its score counts as a pass.

    prompt selects the template: "strict" (fine-grained consistency scoring) or
    "topic" (only catches wrong-document answers, ignores detail differences).
    """

    client: OpenAI
    model: str
    prompt: str = DEFAULT_PROMPT
    temperature: float = 0.0
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    retries: int = DEFAULT_JUDGE_RETRIES
    threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.prompt not in _PROMPTS:
            raise ValueError(f"Unknown judge prompt {self.prompt!r}. Choose from: {PROMPT_NAMES}")


def _parse_judge_response(raw: str, threshold: float) -> EvaluationResult:
    """Return the verdict of a judge response; raise ValueError naming the problem when it has none.

    Only a complete JSON object with an integer score in 0-10 and a string reasoning counts.
    The score alone decides the verdict; a "verdict" field in the response is ignored.
    """
    if not raw:
        raise ValueError("empty response")

    try:
        data = json.loads(_extract_json(_strip_reasoning(raw)))
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON ({exc.msg})") from exc

    if not isinstance(data, dict):
        raise ValueError(f"JSON is not an object: {type(data).__name__}")

    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, int):
        raise ValueError(f"score is not an integer: {score!r}")
    if not 0 <= score <= 10:
        raise ValueError(f"score out of range 0-10: {score}")

    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str):
        raise ValueError(f"reasoning is not a string: {reasoning!r}")

    # For the topic prompt, prepend the identified topics to make failures
    # immediately readable: "Ancient Rome → World War II: ..."
    if "reference_topic" in data and "target_topic" in data:
        reasoning = f"[{data['reference_topic']} → {data['target_topic']}] {reasoning}"

    score_01 = score / 10.0
    return EvaluationResult(score=round(score_01, 3), reasoning=reasoning, passed=score_01 >= threshold)


def evaluate_answers(
    question: str,
    target_answer: str,
    reference_answer: str,
    judge: JudgeConfig,
) -> EvaluationTrace:
    """Compare target_answer to reference_answer using LLM-as-judge.

    The evaluation call itself uses a unique prefix to prevent cache hits,
    since the judge must not be influenced by prior cached state either.

    A judge response without a valid verdict (empty, truncated, not JSON, wrong schema)
    says nothing about the target answer, so the call is repeated up to judge.retries
    times. A valid verdict, PASS or FAIL, is never retried. Every attempt gets a fresh
    prefix, which also changes the generation at temperature 0. When no attempt yields
    a valid verdict the trace carries judge_error and a non-passing result.

    Returns an EvaluationTrace that bundles the EvaluationResult with the full
    judge message exchange, the raw response and every attempt for logging.
    """
    prompt = _PROMPTS[judge.prompt].format(
        question=question,
        reference_answer=reference_answer,
        target_answer=target_answer,
    )
    max_attempts = max(judge.retries, 0) + 1
    attempt_logs: list[JudgeAttemptLog] = []
    result: EvaluationResult | None = None

    for attempt in range(1, max_attempts + 1):
        eval_prefix = str(uuid.uuid4())
        messages = [
            {
                "role": "system",
                "content": (
                    f"[eval-session:{eval_prefix}] "
                    "You are a precise answer evaluator. Respond only with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = judge.client.chat.completions.create(
            model=judge.model,
            messages=messages,
            max_tokens=judge.max_tokens,
            temperature=judge.temperature,
        )
        choice = response.choices[0]
        raw = (choice.message.content or "").strip()

        problem: str | None = None
        try:
            result = _parse_judge_response(raw, judge.threshold)
        except ValueError as exc:
            problem = str(exc)

        attempt_logs.append(JudgeAttemptLog(
            prefix=eval_prefix,
            raw_response=raw,
            finish_reason=choice.finish_reason or "",
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            request_id=response.id or "",
            problem=problem,
        ))
        if problem is None:
            break

        outcome = "Retrying…" if attempt < max_attempts else "Giving up."
        _stderr.print(
            f"[yellow]WARNING[/yellow]: no valid judge verdict on attempt {attempt}/{max_attempts}: "
            f"{escape(problem)} (finish_reason={choice.finish_reason!r}, request_id={response.id}). {outcome}"
        )

    judge_error = attempt_logs[-1].problem
    if judge_error is not None:
        result = EvaluationResult(
            score=0.0,
            reasoning=f"[judge error after {attempt} attempt(s)] {judge_error}",
            passed=False,
        )
    elif attempt > 1:
        result.reasoning = f"[judge attempt {attempt}/{max_attempts}] {result.reasoning}"

    return EvaluationTrace(
        result=result,
        judge_messages=messages,
        judge_raw_response=raw,
        eval_prefix=eval_prefix,
        judge_attempts=attempt,
        attempt_logs=attempt_logs,
        judge_error=judge_error,
    )
