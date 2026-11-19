from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import httpx
import pytest


def make_response(content: str, finish_reason: str = "stop", request_id: str = "req-1") -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(completion_tokens=len(content.split()), completion_tokens_details=None)
    return SimpleNamespace(id=request_id, choices=[choice], usage=usage)


def make_request() -> httpx.Request:
    return httpx.Request("POST", "http://server/v1/chat/completions")


class FakeCompletions:
    def __init__(self, responder: Callable[[dict], SimpleNamespace]) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responder(kwargs)


class FakeClient:
    """Stands in for openai.OpenAI: `responder` gets the request kwargs and returns or raises."""

    def __init__(self, responder: Callable[[dict], SimpleNamespace]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responder))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


@pytest.fixture
def fake_client() -> type[FakeClient]:
    return FakeClient
