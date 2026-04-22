import uuid
from openai import OpenAI
from kvcache_sanity.models import Scenario, Document

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
) -> str:
    messages = build_messages(scenario, documents, unique_prefix)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def get_reference_answer(
    scenario: Scenario,
    documents: dict[str, Document],
    client: OpenAI,
    model: str,
    max_tokens: int = 512,
) -> str:
    """Run the scenario with a unique prefix to force a full KV cache miss.

    The UUID prefix is placed at the start of the system message. Because KV
    caches are keyed on the full token prefix, any difference in the first token
    guarantees that no cached blocks are reused for this call.
    """
    prefix = str(uuid.uuid4())
    return run_scenario(scenario, documents, client, model, prefix, max_tokens)
