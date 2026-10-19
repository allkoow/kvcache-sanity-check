import json
import re
import uuid
from openai import OpenAI
from rich.console import Console as _Console
from kvcache_sanity.models import EvaluationResult, EvaluationTrace

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
DEFAULT_JUDGE_MAX_TOKENS = 1024
DEFAULT_JUDGE_RETRIES = 2


def _parse_judge_response(raw: str, threshold: float) -> EvaluationResult | None:
    """Return the verdict carried by a judge response, or None when it contains no score."""
    answer = _strip_reasoning(raw)

    try:
        data = json.loads(_extract_json(answer))
        score_01 = float(data["score"]) / 10.0
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # Truncated JSON fallback: extract score with regex if present
        score_match = re.search(r'"score"\s*:\s*(\d+)', answer)
        if not score_match:
            return None
        verdict_match = re.search(r'"verdict"\s*:\s*"(pass|fail)"', answer, re.IGNORECASE)
        score_01 = float(score_match.group(1)) / 10.0
        passed = verdict_match.group(1).lower() == "pass" if verdict_match else score_01 >= threshold
        return EvaluationResult(
            score=round(score_01, 3),
            reasoning=f"[truncated JSON, score extracted] {answer[:200]}",
            passed=passed,
        )

    # For the topic prompt, prepend the identified topics to make failures
    # immediately readable: "Ancient Rome → World War II: ..."
    reasoning = str(data.get("reasoning", ""))
    if "reference_topic" in data and "target_topic" in data:
        reasoning = f"[{data['reference_topic']} → {data['target_topic']}] {reasoning}"
    return EvaluationResult(score=round(score_01, 3), reasoning=reasoning, passed=score_01 >= threshold)


def _unparsed_result(raw: str) -> EvaluationResult:
    if not raw:
        return EvaluationResult(
            score=0.0,
            reasoning="[empty response from judge — server returned no content]",
            passed=False,
        )

    passed = "pass" in raw.lower() and "fail" not in raw.lower()
    return EvaluationResult(
        score=1.0 if passed else 0.0,
        reasoning=f"[JSON parse failed] raw response: {raw[:300]}",
        passed=passed,
    )


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
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    judge_retries: int = DEFAULT_JUDGE_RETRIES,
) -> EvaluationTrace:
    """Compare target_answer to reference_answer using LLM-as-judge.

    judge_prompt selects which prompt template to use: "strict" (default,
    fine-grained consistency scoring) or "topic" (only catches wrong-document
    answers, ignores detail differences).

    The evaluation call itself uses a unique prefix to prevent cache hits,
    since the judge must not be influenced by prior cached state either.

    A judge response without a score (empty, truncated mid-reasoning, no JSON)
    says nothing about the target answer, so the call is repeated up to
    judge_retries times. Every attempt gets a fresh prefix, which also changes
    the generation at temperature 0.

    Returns an EvaluationTrace that bundles the EvaluationResult with the full
    judge message exchange and raw response for logging.
    """
    if judge_prompt not in _PROMPTS:
        raise ValueError(f"Unknown judge_prompt {judge_prompt!r}. Choose from: {PROMPT_NAMES}")

    eval_client = judge_client or client
    prompt = _PROMPTS[judge_prompt].format(
        question=question,
        reference_answer=reference_answer,
        target_answer=target_answer,
    )
    max_attempts = max(judge_retries, 0) + 1

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
        response = eval_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=judge_max_tokens,
            temperature=judge_temperature,
        )
        choice = response.choices[0]
        raw = (choice.message.content or "").strip()

        result = _parse_judge_response(raw, threshold)
        if result is not None:
            break

        if attempt < max_attempts:
            _stderr.print(
                f"[yellow]WARNING[/yellow]: judge response has no score on attempt {attempt}/{max_attempts} "
                f"(finish_reason={choice.finish_reason!r}, request_id={response.id}). Retrying…"
            )
    else:
        result = _unparsed_result(raw)

    if attempt > 1:
        result.reasoning = f"[judge attempt {attempt}/{max_attempts}] {result.reasoning}"

    return EvaluationTrace(
        result=result,
        judge_messages=messages,
        judge_raw_response=raw,
        eval_prefix=eval_prefix,
        judge_attempts=attempt,
    )
