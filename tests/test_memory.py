"""Scenario 7: session memory persists and isolates conversations correctly."""

from orchestrator.memory import SessionMemory


def test_memory_persists_across_turns_within_a_session():
    memory = SessionMemory()

    memory.append("session-a", "user", "hi")
    memory.append("session-a", "assistant", "hello, how can I help?")
    memory.append("session-a", "user", "what's my order status")

    history = memory.as_api_messages("session-a")

    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[0]["content"] == "hi"


def test_memory_isolates_different_sessions():
    memory = SessionMemory()

    memory.append("session-a", "user", "message in session a")
    memory.append("session-b", "user", "message in session b")

    assert len(memory.get("session-a")) == 1
    assert len(memory.get("session-b")) == 1
    assert memory.get("session-a")[0].content != memory.get("session-b")[0].content


def test_memory_clear_removes_session():
    memory = SessionMemory()
    memory.append("session-a", "user", "hi")

    memory.clear("session-a")

    assert memory.get("session-a") == []
