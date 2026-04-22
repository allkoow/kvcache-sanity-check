import uuid
from openai import OpenAI
from kvcache_sanity.models import Scenario, ScenarioPair, Document, RunResult

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
        "I will provide several documents for you to read, then ask a question about them."
    )
    if unique_prefix:
        system_content = f"[{unique_prefix}] " + system_content

    messages: list[dict] = [{"role": "system", "content": system_content}]

    for i, doc_id in enumerate(scenario.document_ids, 1):
        doc = documents[doc_id]
        messages.append({
            "role": "user",
            "content": f"Document {i} — {doc.title}:\n\n{doc.content}",
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
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return RunResult(
        answer=response.choices[0].message.content or "",
        messages=messages,
        unique_prefix=unique_prefix,
    )


def get_reference_answer(
    scenario: Scenario,
    documents: dict[str, Document],
    client: OpenAI,
    model: str,
    max_tokens: int = 512,
) -> RunResult:
    """Run the scenario with a unique prefix to force a full KV cache miss.

    The UUID prefix is placed at the start of the system message. Because KV
    caches are keyed on the full token prefix, any difference in the first token
    guarantees that no cached blocks are reused for this call.
    """
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
    system_content = "You are a helpful assistant."
    if unique_prefix:
        system_content = f"[{unique_prefix}] " + system_content

    messages: list[dict] = [{"role": "system", "content": system_content}]

    for i, pair in enumerate(pairs[:up_to_index + 1]):
        doc = documents[pair.doc_id]
        messages.append({
            "role": "user",
            "content": f"{doc.content}\n\n{pair.question}",
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
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return RunResult(
        answer=response.choices[0].message.content or "",
        messages=messages,
        unique_prefix=unique_prefix,
    )


def get_reference_pair_answer(
    pairs: list[ScenarioPair],
    documents: dict[str, Document],
    client: OpenAI,
    model: str,
    pair_index: int,
    max_tokens: int = 512,
) -> RunResult:
    """Same as run_sequential_pair but with a UUID prefix to force a cache miss."""
    prefix = str(uuid.uuid4())
    return run_sequential_pair(pairs, documents, client, model, pair_index, prefix, max_tokens)
