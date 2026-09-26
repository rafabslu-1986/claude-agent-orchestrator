"""Etapa 6: the FastAPI wrapper (api/main.py), tested the same way as the
rest of the suite -- fully offline, against FakeClaudeClient, never against
the real Anthropic API. This proves the HTTP layer wires requests into
`Orchestrator.handle_message` correctly; it is not a live-server test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient

from conftest import FakeClaudeClient, text_response, tool_use_response

import api.main as api_main
from orchestrator.orchestrator import Orchestrator


def _client_with(orchestrator: Orchestrator) -> TestClient:
    # Swap the module-level singleton the app closures over, same trick the
    # rest of the project uses to keep tests offline and deterministic.
    api_main._orchestrator = orchestrator
    return TestClient(api_main.app)


def test_health_endpoint():
    client = _client_with(Orchestrator(client=FakeClaudeClient([])))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_post_message_routes_to_a_specialist_and_returns_the_reply():
    fake = FakeClaudeClient(
        [
            text_response("sales"),
            text_response("Sure — could you tell me roughly how many seats you need?"),
        ]
    )
    client = _client_with(Orchestrator(client=fake))

    response = client.post(
        "/messages", json={"session_id": "api-session-1", "message": "Tell me about pricing"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "sales"
    assert body["escalated"] is False
    assert body["tool_calls_made"] == []
    assert "seats" in body["text"]


def test_post_message_grounds_the_answer_via_the_knowledge_base_tool():
    fake = FakeClaudeClient(
        [
            text_response("billing"),
            tool_use_response("search_knowledge_base", {"query": "refund timeline"}),
            text_response("Refunds land back on your card within 5 to 10 business days."),
        ]
    )
    client = _client_with(Orchestrator(client=fake))

    response = client.post(
        "/messages",
        json={"session_id": "api-session-2", "message": "When do I get refunded?"},
    )

    body = response.json()
    assert body["tool_calls_made"] == ["search_knowledge_base"]
    assert "5 to 10 business days" in body["text"]


def test_post_message_rejects_an_empty_message_without_calling_the_model():
    fake = FakeClaudeClient([])  # would raise "ran out of scripted responses" if ever called
    client = _client_with(Orchestrator(client=fake))

    response = client.post("/messages", json={"session_id": "api-session-3", "message": "   "})

    assert response.status_code == 422
    assert fake.calls == []


def test_export_session_returns_the_full_conversation():
    fake = FakeClaudeClient(
        [
            text_response("sales"),
            text_response("Sure — could you tell me roughly how many seats you need?"),
        ]
    )
    client = _client_with(Orchestrator(client=fake))
    client.post("/messages", json={"session_id": "api-session-4", "message": "Growth plan info"})

    response = client.get("/sessions/api-session-4")

    assert response.json() == [
        {"role": "user", "content": "Growth plan info"},
        {"role": "assistant", "content": "Sure — could you tell me roughly how many seats you need?"},
    ]


def test_forget_session_erases_the_conversation():
    fake = FakeClaudeClient(
        [
            text_response("sales"),
            text_response("Sure — could you tell me roughly how many seats you need?"),
        ]
    )
    client = _client_with(Orchestrator(client=fake))
    client.post("/messages", json={"session_id": "api-session-5", "message": "Growth plan info"})

    delete_response = client.delete("/sessions/api-session-5")
    export_response = client.get("/sessions/api-session-5")

    assert delete_response.status_code == 204
    assert export_response.json() == []


def test_second_message_is_escalated_via_the_api_once_the_session_budget_is_exceeded():
    fake = FakeClaudeClient(
        [
            text_response("sales", usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000}),
            text_response(
                "Sure — could you tell me roughly how many seats you need?",
                usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000},
            ),
        ]
    )
    client = _client_with(Orchestrator(client=fake, max_session_cost_usd=0.01))

    first = client.post(
        "/messages", json={"session_id": "api-session-6", "message": "Growth plan info"}
    )
    second = client.post(
        "/messages", json={"session_id": "api-session-6", "message": "one more question"}
    )

    assert first.json()["escalated"] is False
    assert second.json()["escalated"] is True
    assert second.json()["agent_name"] == "budget_guardrail"
    assert len(fake.calls) == 2  # unchanged -- the second turn never called the model


# --- Etapa 15: X-API-Key auth, opt-in via ORCHESTRATOR_API_KEY -------------


def test_messages_endpoint_is_open_by_default_when_no_key_is_configured(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_API_KEY", raising=False)
    fake = FakeClaudeClient([text_response("sales"), text_response("Sure — how many seats?")])
    client = _client_with(Orchestrator(client=fake))

    response = client.post(
        "/messages", json={"session_id": "auth-session-1", "message": "pricing please"}
    )

    assert response.status_code == 200


def test_messages_endpoint_rejects_requests_with_no_key_once_configured(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_API_KEY", "s3cr3t")
    fake = FakeClaudeClient([])  # never reached -- rejected before the model is called
    client = _client_with(Orchestrator(client=fake))

    response = client.post(
        "/messages", json={"session_id": "auth-session-2", "message": "pricing please"}
    )

    assert response.status_code == 401
    assert fake.calls == []


def test_messages_endpoint_rejects_the_wrong_key(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_API_KEY", "s3cr3t")
    fake = FakeClaudeClient([])
    client = _client_with(Orchestrator(client=fake))

    response = client.post(
        "/messages",
        json={"session_id": "auth-session-3", "message": "pricing please"},
        headers={"X-API-Key": "wrong"},
    )

    assert response.status_code == 401
    assert fake.calls == []


def test_messages_endpoint_accepts_the_correct_key(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_API_KEY", "s3cr3t")
    fake = FakeClaudeClient([text_response("sales"), text_response("Sure — how many seats?")])
    client = _client_with(Orchestrator(client=fake))

    response = client.post(
        "/messages",
        json={"session_id": "auth-session-4", "message": "pricing please"},
        headers={"X-API-Key": "s3cr3t"},
    )

    assert response.status_code == 200


def test_session_export_and_delete_are_also_gated_once_a_key_is_configured(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_API_KEY", "s3cr3t")
    fake = FakeClaudeClient([])
    client = _client_with(Orchestrator(client=fake))

    export_without_key = client.get("/sessions/auth-session-5")
    delete_without_key = client.delete("/sessions/auth-session-5")
    export_with_key = client.get(
        "/sessions/auth-session-5", headers={"X-API-Key": "s3cr3t"}
    )

    assert export_without_key.status_code == 401
    assert delete_without_key.status_code == 401
    assert export_with_key.status_code == 200


def test_health_endpoint_stays_open_even_when_a_key_is_configured(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_API_KEY", "s3cr3t")
    client = _client_with(Orchestrator(client=FakeClaudeClient([])))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
