import uuid
from openai import OpenAI
from rich.console import Console as _Console
from kvcache_sanity.models import Scenario, ScenarioPair, Document, RunResult

_stderr = _Console(stderr=True)
_MAX_EMPTY_RETRIES = 3


def _call_api(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    label: str = "",
) -> tuple[str, str, int]:
    """Call the chat completions API, retrying on empty answers and warning on truncation.

    Returns (answer, finish_reason, completion_tokens).
    """
    tag = f" [{label}]" if label else ""
    answer = ""
    finish_reason = ""
    completion_tokens = 0

    for attempt in range(_MAX_EMPTY_RETRIES + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        choice = response.choices[0]
        answer = choice.message.content or ""
        finish_reason = choice.finish_reason or ""
        completion_tokens = (response.usage.completion_tokens if response.usage else 0)

        if finish_reason == "length":
            _stderr.print(
                f"[yellow]WARNING[/yellow]: response truncated at {max_tokens} tokens "
                f"(completion_tokens={completion_tokens}){tag}. "
                "Consider increasing --max-tokens."
            )

        if answer.strip():
            return answer, finish_reason, completion_tokens

        if attempt < _MAX_EMPTY_RETRIES:
            _stderr.print(
                f"[yellow]WARNING[/yellow]: empty answer on attempt {attempt + 1}/{_MAX_EMPTY_RETRIES}{tag}. "
                f"finish_reason={finish_reason!r}, completion_tokens={completion_tokens}. Retrying…"
            )
        else:
            _stderr.print(
                f"[red]WARNING[/red]: empty answer after {_MAX_EMPTY_RETRIES} retries{tag}. "
                f"finish_reason={finish_reason!r}, completion_tokens={completion_tokens}. Giving up."
            )

    return answer, finish_reason, completion_tokens

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
    max_tokens: int = 512,
) -> RunResult:
    messages = build_messages(scenario, documents, unique_prefix)
    label = f"scenario={scenario.id}, prefix={'ref' if unique_prefix else 'target'}"
    answer, finish_reason, completion_tokens = _call_api(client, model, messages, max_tokens, label)
    return RunResult(
        answer=answer,
        messages=messages,
        unique_prefix=unique_prefix,
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
    )


def get_reference_answer(
    scenario: Scenario,
    documents: dict[str, Document],
    client: OpenAI,
    model: str,
    max_tokens: int = 512,
    prefix: str | None = None,
) -> RunResult:
    """Run the scenario with a cache-busting prefix to force a full KV cache miss.

    If prefix is supplied it is reused (so subsequent target calls with the same
    prefix will hit blocks cached by this reference call). If omitted a fresh UUID
    is generated, which is appropriate when the caller wants an isolated ground truth.
    """
    if prefix is None:
        prefix = str(uuid.uuid4())
    return run_scenario(scenario, documents, client, model, prefix, max_tokens)


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
    max_tokens: int = 512,
) -> RunResult:
    """Run the conversation up to pair_index and return the answer for that pair."""
    messages = build_pairs_messages(pairs, documents, pair_index, unique_prefix)
    label = f"pair={pair_index}, prefix={'ref' if unique_prefix else 'target'}"
    answer, finish_reason, completion_tokens = _call_api(client, model, messages, max_tokens, label)
    return RunResult(
        answer=answer,
        messages=messages,
        unique_prefix=unique_prefix,
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
    )


def get_reference_pair_answer(
    pairs: list[ScenarioPair],
    documents: dict[str, Document],
    client: OpenAI,
    model: str,
    pair_index: int,
    max_tokens: int = 512,
    prefix: str | None = None,
) -> RunResult:
    """Same as run_sequential_pair but with a cache-busting prefix to force a cache miss.

    If prefix is supplied it is reused so subsequent target calls with the same
    prefix will hit blocks cached by this reference call.
    """
    if prefix is None:
        prefix = str(uuid.uuid4())
    return run_sequential_pair(pairs, documents, client, model, pair_index, prefix, max_tokens)
