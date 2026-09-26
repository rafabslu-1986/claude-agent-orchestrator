"""Shared test fixtures.

FakeClaudeClient replaces ClaudeClient entirely: it never imports the
anthropic package and never makes a network call. Each test queues up the
exact sequence of ClaudeResponse objects it wants returned, call by call,
so the orchestration logic (routing, tool-use loop, escalation, memory) is
tested deterministically instead of depending on live model output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from orchestrator.claude_client import ClaudeResponse


class FakeClaudeClient:
    """Each entry in scripted_responses is either a ClaudeResponse (returned
    normally) or a BaseException instance (raised instead) -- the latter is
    what Etapa 13's resilience tests use to script a flaky API without
    touching the real SDK or network.
    """

    def __init__(self, scripted_responses: list[ClaudeResponse | BaseException]):
        self._responses = list(scripted_responses)
        self.calls: list[dict] = []

    def send(self, system, messages, tools=None, max_tokens=1024, temperature=0.2):
        self.calls.append(
            {"system": system, "messages": messages, "tools": tools, "max_tokens": max_tokens}
        )
        if not self._responses:
            raise AssertionError("FakeClaudeClient ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def knowledge_base_dir():
    return Path(__file__).parent.parent / "src" / "orchestrator" / "knowledge_base"


def text_response(text: str) -> ClaudeResponse:
    return ClaudeResponse(text=text, tool_calls=[], stop_reason="end_turn")


def tool_use_response(name: str, input_: dict, call_id: str = "call_1") -> ClaudeResponse:
    return ClaudeResponse(
        text="",
        tool_calls=[{"id": call_id, "name": name, "input": input_}],
        stop_reason="tool_use",
    )


def fake_api_error(status_code: int, message: str = "boom"):
    """Builds the exact exception instance the real anthropic SDK would
    raise for this HTTP status code -- reuses the SDK's own
    status-code-to-exception-class mapping (Anthropic._make_status_error)
    instead of a hand-rolled subset, so Etapa 13's resilience tests
    classify the real exception types, not stand-ins that happen to share a
    name.
    """
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request, json={"error": {"message": message}})
    dummy_client = anthropic.Anthropic(api_key="sk-ant-test-not-a-real-key")
    return dummy_client._make_status_error(message, body=None, response=response)


def fake_connection_error(timeout: bool = False):
    """APITimeoutError (a subclass of APIConnectionError) if timeout=True,
    otherwise a plain APIConnectionError -- both transient, neither has an
    HTTP status code."""
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if timeout:
        return anthropic.APITimeoutError(request=request)
    return anthropic.APIConnectionError(request=request)
