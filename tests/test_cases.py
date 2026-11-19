from __future__ import annotations

from kvcache_sanity.cases import build_cases, messages_sha256
from kvcache_sanity.corpus import load_documents
from kvcache_sanity.runner import build_messages, inject_prefix
from kvcache_sanity.scenarios import load_scenarios


def _scenario(scenario_id: str):
    return next(item for item in load_scenarios(None) if item.id == scenario_id)


def test_inject_prefix_matches_build_messages():
    documents = load_documents(None)
    scenario = _scenario("early_doc_recall")

    assert inject_prefix(build_messages(scenario, documents, None), "abc") == build_messages(scenario, documents, "abc")


def test_inject_prefix_leaves_input_untouched():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]

    prefixed = inject_prefix(messages, "abc")

    assert prefixed[0]["content"] == "[abc] sys"
    assert messages[0]["content"] == "sys"
    assert inject_prefix(messages, None) == messages
    assert inject_prefix(messages, None) is not messages


def test_build_cases_skips_sequential_pairs_and_hashes_messages():
    cases = build_cases(load_scenarios(None), load_documents(None))
    ids = [case.case_id for case in cases]

    assert "sequential_summarize" not in ids
    assert "early_doc_recall" in ids
    assert all(len(case.messages_sha256) == 64 for case in cases)
    assert all(case.messages_sha256 == messages_sha256(case.messages) for case in cases)
    assert not cases[0].messages[0]["content"].startswith("[")


def test_case_carries_scenario_temperature_override():
    scenario = _scenario("early_doc_recall").model_copy(update={"temperature": 0.7})

    (case,) = build_cases([scenario], load_documents(None))

    assert case.temperature == 0.7
    assert case.question == scenario.question
