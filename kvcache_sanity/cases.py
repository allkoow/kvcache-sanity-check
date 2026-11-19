from __future__ import annotations

import hashlib
import json

from kvcache_sanity.models import Case, Document, Scenario
from kvcache_sanity.runner import build_messages


def messages_sha256(messages: list[dict]) -> str:
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)

    return hashlib.sha256(serialized.encode()).hexdigest()


def build_cases(scenarios: list[Scenario], documents: dict[str, Document]) -> list[Case]:
    """One case per multi_turn_recall scenario; sequential_pairs scenarios are not phase-aware."""

    cases = []

    for scenario in scenarios:
        if scenario.mode == "sequential_pairs":
            continue

        messages = build_messages(scenario, documents)
        case = Case(
            case_id=scenario.id,
            scenario_id=scenario.id,
            question=scenario.question,
            messages=messages,
            messages_sha256=messages_sha256(messages),
            temperature=scenario.temperature,
        )
        cases.append(case)

    return cases
