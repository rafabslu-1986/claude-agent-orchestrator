"""Scenario 1-2: router classification, including the unknown fallback."""

from conftest import FakeClaudeClient, text_response

from orchestrator.router import Router


def test_router_classifies_known_intent():
    client = FakeClaudeClient([text_response("billing")])
    router = Router(client)

    result = router.classify("Why was I charged twice this month?")

    assert result.intent == "billing"


def test_router_falls_back_to_unknown_on_unexpected_output():
    # Claude answering with something outside the fixed vocabulary must never
    # crash the router — it must degrade to "unknown" so the orchestrator hands
    # off to a human instead of silently misrouting.
    client = FakeClaudeClient([text_response("i think this is about billing maybe")])
    router = Router(client)

    result = router.classify("something ambiguous")

    assert result.intent == "unknown"
