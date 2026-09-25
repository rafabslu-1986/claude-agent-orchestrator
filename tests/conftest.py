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
    def __init__(self, scripted_responses: list[ClaudeResponse]):
        self._responses = list(scripted_responses)
        self.calls: list[dict] = []

    def send(self, system, messages, tools=None, max_tokens=1024, temperature=0.2):
        self.calls.append(
            {"system": system, "messages": messages, "tools": tools, "max_tokens": max_tokens}
        )
        if not self._responses:
            raise AssertionError("FakeClaudeClient ran out of scripted responses")
        return self._responses.pop(0)


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
