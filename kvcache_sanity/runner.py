import re
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from openai import OpenAI
from rich.console import Console as _Console
from rich.markup import escape
from kvcache_sanity.models import Scenario, ScenarioPair, Document, RunResult

_stderr = _Console(stderr=True)
_MAX_EMPTY_RETRIES = 3
DEFAULT_REFERENCE_RETRIES = 2

# Loop detection looks at the tail of a truncated output: in a degenerate loop most
# fixed-length substrings of the tail occur more than once, in normal prose almost none do.
_LOOP_TAIL_CHARS = 2000
_LOOP_MIN_TAIL_CHARS = 200
_LOOP_SHINGLE_CHARS = 32
_LOOP_REPEATED_RATIO = 0.5
_DIGITS = re.compile(r"\d")


def repeated_tail_ratio(text: str) -> float:
    """Fraction of 32-char substrings in the output tail that already appeared earlier in that tail.

    Digits are masked so loops that only advance a counter ("Step 7: …", "Step 8: …") still count.
    """

    tail = _DIGITS.sub("0", " ".join(text.split()))[-_LOOP_TAIL_CHARS:]

    if len(tail) < _LOOP_MIN_TAIL_CHARS:
        return 0.0

    total = len(tail) - _LOOP_SHINGLE_CHARS + 1
    unique = {tail[i:i + _LOOP_SHINGLE_CHARS] for i in range(total)}

    return 1 - len(unique) / total


def classify_truncation(text: str) -> str:
    """'loop' when a truncated output is repeating itself (more tokens would not help), else 'budget'."""
    return "loop" if repeated_tail_ratio(text) >= _LOOP_REPEATED_RATIO else "budget"


def _truncated_in_reasoning(content: str, reasoning: str) -> bool:
    """True when generation stopped before the final answer began.

    Detectable only when the server returns reasoning in a separate field.
    """
    return bool(reasoning.strip()) and not content.strip()


def describe_generation(run: RunResult) -> str:
    """One-line summary of how a generation ended, for warnings and incomplete-case reasons."""

    parts = [f"finish_reason={run.finish_reason!r}"]
    if run.truncation:
        parts.append(f"truncation={run.truncation}")

    if run.truncated_in_reasoning is not None:
        parts.append(f"truncated_in_reasoning={run.truncated_in_reasoning}")

    parts.append(f"completion_tokens={run.completion_tokens}")

    if run.reasoning_tokens is not None:
        parts.append(f"reasoning_tokens={run.reasoning_tokens}")

    parts.append(f"answer_chars={len(run.answer.strip())}")
    parts.append(f"request_id={run.request_id}")

    return ", ".join(parts)


def _call_api(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    label: str = "",
    temperature: float = 0.1,
    empty_retries: int = _MAX_EMPTY_RETRIES,
) -> RunResult:
    """Call the chat completions API, retrying on empty answers and warning on truncation.

    request_time is an ISO-8601 UTC timestamp captured immediately after the response
    is received — useful for correlating with server-side logs.
    """
    tag = escape(f" [{label}]") if label else ""
    run = RunResult(answer="", messages=messages)

    for attempt in range(empty_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
        reasoning = reasoning if isinstance(reasoning, str) else ""
        run = RunResult(
            answer=choice.message.content or "",
            messages=messages,
            finish_reason=choice.finish_reason or "",
            completion_tokens=usage.completion_tokens if usage else 0,
            request_id=response.id or "",
            request_time=datetime.now(timezone.utc).isoformat(),
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
        )

        if run.finish_reason == "length":
            run.truncation = classify_truncation(reasoning + "\n" + run.answer)
            run.truncated_in_reasoning = _truncated_in_reasoning(run.answer, reasoning)
            advice = (
                "Output is repeating itself; more tokens will not help."
                if run.truncation == "loop"
                else "Consider increasing --max-tokens."
            )
            _stderr.print(
                f"[yellow]WARNING[/yellow]: response truncated at {max_tokens} tokens "
                f"({describe_generation(run)}){tag}. {advice}"
            )

        if run.answer.strip():
            return run

        if attempt < empty_retries:
            _stderr.print(
                f"[yellow]WARNING[/yellow]: empty answer on attempt {attempt + 1}/{empty_retries}{tag}. "
                f"{describe_generation(run)}. Retrying…"
            )
        elif empty_retries > 0:
            _stderr.print(
                f"[red]WARNING[/red]: empty answer after {empty_retries} retries{tag}. "
                f"{describe_generation(run)}. Giving up."
            )

    return run

# Short hardcoded acknowledgment so conversation structure is identical
# between target and reference runs. This ensures the only variable is
# whether the KV cache is hit (controlled by the system message prefix).
_ACK = "Understood, I've read that document. Ready for the next one or your question."


def build_messages(
    scenario: Scenario,
    documents: dict[str, Document],
    unique_prefix: str | None = None,
) -> list[dict]:
    """Build the multi-turn conversation for a scenario.

    Each document gets its own user turn with a short assistant acknowledgment,
    so the KV cache grows incrementally as in real chat usage. The final user
    turn is the question.

    If unique_prefix is set it is injected into the system message, guaranteeing
    a cache miss on any KV cache keyed on the full conversation prefix.
    """
    system_content = (
        "You are a helpful assistant. "
        "I will provide several documents for you to read, then ask a question about them. "
        "Each document is enclosed in <document>...</document> tags. "
        "When asked about a specific document, answer ONLY about that document. "
        "Do not summarize or mention other documents unless explicitly asked."
    )
    if unique_prefix:
        system_content = f"[{unique_prefix}] " + system_content

    messages: list[dict] = [{"role": "system", "content": system_content}]

    for i, doc_id in enumerate(scenario.document_ids, 1):
        doc = documents[doc_id]
        messages.append({
            "role": "user",
            "content": f"Document {i} — {doc.title}:\n<document>\n{doc.content}\n</document>",
        })
        messages.append({"role": "assistant", "content": _ACK})

    messages.append({"role": "user", "content": scenario.question})
    return messages


def run_scenario(
    scenario: Scenario,
    documents: dict[str, Document],
    client: OpenAI,
    model: str,
    unique_prefix: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    empty_retries: int = _MAX_EMPTY_RETRIES,
) -> RunResult:
    messages = build_messages(scenario, documents, unique_prefix)
    label = f"scenario={scenario.id}, prefix={'ref' if unique_prefix else 'target'}"
    run = _call_api(client, model, messages, max_tokens, label, temperature=temperature, empty_retries=empty_retries)
    run.unique_prefix = unique_prefix
    return run


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

    Empty answers are not retried here: the prefix is already cached after the first call,
    so a retry under it would not be a clean recompute. Use prepare_reference instead.
    """
    if prefix is None:
        prefix = str(uuid.uuid4())
    return run_scenario(
        scenario, documents, client, model, prefix, max_tokens, temperature=temperature, empty_retries=0,
    )


# ---------------------------------------------------------------------------
# sequential_pairs mode
# ---------------------------------------------------------------------------

_PAIRS_ACK = "Got it."


def build_pairs_messages(
    pairs: list[ScenarioPair],
    documents: dict[str, Document],
    up_to_index: int,
    unique_prefix: str | None = None,
) -> list[dict]:
    """Build a multi-turn conversation for sequential_pairs mode.

    Simulates a user who pastes documents with questions into the same chat
    session without ever starting a fresh conversation — each earlier pair
    primes the KV cache. The conversation is truncated after the question at
    up_to_index so the caller can capture that turn's answer.

    Earlier assistant turns use a short hardcoded ACK so the conversation
    structure is identical between target and reference runs.
    """
    system_content = (
        "You are a helpful assistant. "
        "The user will paste text enclosed in <document>...</document> tags followed by a question. "
        "Answer ONLY about the document in that specific message. "
        "Do not refer to, summarize, or include content from documents in any previous messages."
    )
    if unique_prefix:
        system_content = f"[{unique_prefix}] " + system_content

    messages: list[dict] = [{"role": "system", "content": system_content}]

    for i, pair in enumerate(pairs[:up_to_index + 1]):
        doc = documents[pair.doc_id]
        # Lead with the question so the model knows what to look for before
        # reading the document, and the boundary is unambiguous.
        q = pair.question[0].lower() + pair.question[1:]  # merge into preamble
        messages.append({
            "role": "user",
            "content": f"Please read the document below and {q}\n\n<document>\n{doc.content}\n</document>",
        })
        if i < up_to_index:
            messages.append({"role": "assistant", "content": _PAIRS_ACK})

    return messages


def run_sequential_pair(
    pairs: list[ScenarioPair],
    documents: dict[str, Document],
    client: OpenAI,
    model: str,
    pair_index: int,
    unique_prefix: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    empty_retries: int = _MAX_EMPTY_RETRIES,
) -> RunResult:
    """Run the conversation up to pair_index and return the answer for that pair."""
    messages = build_pairs_messages(pairs, documents, pair_index, unique_prefix)
    label = f"pair={pair_index}, prefix={'ref' if unique_prefix else 'target'}"
    run = _call_api(client, model, messages, max_tokens, label, temperature=temperature, empty_retries=empty_retries)
    run.unique_prefix = unique_prefix
    return run


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
    prefix will hit blocks cached by this reference call. Empty answers are not
    retried, see get_reference_answer.
    """
    if prefix is None:
        prefix = str(uuid.uuid4())
    return run_sequential_pair(
        pairs, documents, client, model, pair_index, prefix, max_tokens, temperature=temperature, empty_retries=0,
    )


# ---------------------------------------------------------------------------
# Reference preparation
# ---------------------------------------------------------------------------

def is_complete(run: RunResult) -> bool:
    """A generation is complete when the server stopped on its own and returned a non-empty answer."""
    return run.finish_reason == "stop" and bool(run.answer.strip())


def prepare_reference(
    generate: Callable[[str], RunResult],
    prefix: str,
    retries: int = DEFAULT_REFERENCE_RETRIES,
    label: str = "",
) -> list[RunResult]:
    """Generate the reference, retrying incomplete ones (truncated or empty) under a fresh prefix.

    generate(prefix) runs the reference conversation with the given cache-busting prefix.
    A retry never reuses a prefix: the failed attempt has already cached its blocks, so the
    next call under it would not be a clean recompute.

    Returns every attempt in order. The last one is the reference to use; it is incomplete
    only when all attempts failed.
    """
    tag = escape(f" [{label}]") if label else ""
    max_attempts = max(retries, 0) + 1
    attempts: list[RunResult] = []

    for attempt in range(1, max_attempts + 1):
        run = generate(prefix if attempt == 1 else str(uuid.uuid4()))
        attempts.append(run)
        if is_complete(run):
            break

        outcome = "Retrying with a fresh prefix…" if attempt < max_attempts else "Giving up."
        _stderr.print(
            f"[yellow]WARNING[/yellow]: incomplete reference on attempt {attempt}/{max_attempts}{tag} "
            f"({describe_generation(run)}). {outcome}"
        )

    return attempts
