"""Scenario 7: session memory persists and isolates conversations correctly."""

from datetime import datetime, timedelta, timezone

from orchestrator.memory import Message, SessionMemory


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


def test_memory_without_ttl_never_expires_messages():
    memory = SessionMemory()  # ttl_hours=None: today's behavior, unchanged
    old = datetime.now(timezone.utc) - timedelta(days=365)
    memory._sessions["session-a"] = [Message(role="user", content="ancient", timestamp=old)]

    assert len(memory.get("session-a")) == 1


def test_memory_ttl_prunes_expired_messages_but_keeps_fresh_ones():
    """LGPD Art. 6, III (necessidade): retention is bounded and explicit,
    not indefinite-by-default.
    """
    memory = SessionMemory(ttl_hours=24)
    now = datetime.now(timezone.utc)
    memory._sessions["session-a"] = [
        Message(role="user", content="two days old", timestamp=now - timedelta(days=2)),
        Message(role="assistant", content="one hour old", timestamp=now - timedelta(hours=1)),
    ]

    history = memory.get("session-a")

    assert [m.content for m in history] == ["one hour old"]


def test_memory_ttl_drops_the_session_entirely_once_everything_expires():
    memory = SessionMemory(ttl_hours=24)
    old = datetime.now(timezone.utc) - timedelta(days=2)
    memory._sessions["session-a"] = [Message(role="user", content="stale", timestamp=old)]

    assert memory.get("session-a") == []
    assert "session-a" not in memory._sessions
