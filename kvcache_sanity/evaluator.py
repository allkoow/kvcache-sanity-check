import json
import uuid
from openai import OpenAI
from kvcache_sanity.models import EvaluationResult, EvaluationTrace

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


_JUDGE_PROMPT = """\
You are evaluating whether two answers to the same question are consistent and correct.

Question asked: {question}

Answer A (reference — generated without any cache, treated as ground truth):
{reference_answer}

Answer B (under test — generated using potentially cached KV blocks):
{target_answer}

Evaluate whether Answer B is consistent with Answer A on:
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


def evaluate_answers(
    question: str,
    target_answer: str,
    reference_answer: str,
    client: OpenAI,
    model: str,
    threshold: float = 0.7,
    judge_client: OpenAI | None = None,
) -> EvaluationTrace:
    """Compare target_answer to reference_answer using LLM-as-judge.

    The evaluation call itself uses a unique prefix to prevent cache hits,
    since the judge must not be influenced by prior cached state either.

    If judge_client is provided it is used instead of client for the evaluation
    call (useful when a more capable external model is available for judging).

    Returns an EvaluationTrace that bundles the EvaluationResult with the full
    judge message exchange and raw response for logging.
    """
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
            "content": _JUDGE_PROMPT.format(
                question=question,
                reference_answer=reference_answer,
                target_answer=target_answer,
            ),
        },
    ]

    response = eval_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.0,
    )

    raw = (response.choices[0].message.content or "").strip()

    try:
        data = json.loads(_extract_json(raw))
        score_01 = float(data["score"]) / 10.0
        passed = score_01 >= threshold
        result = EvaluationResult(
            score=round(score_01, 3),
            reasoning=str(data.get("reasoning", "")),
            passed=passed,
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        passed = "pass" in raw.lower() and "fail" not in raw.lower()
        result = EvaluationResult(
            score=1.0 if passed else 0.0,
            reasoning=f"[JSON parse failed] raw response: {raw[:300]}",
            passed=passed,
        )

    return EvaluationTrace(
        result=result,
        judge_messages=messages,
        judge_raw_response=raw,
        eval_prefix=eval_prefix,
    )
