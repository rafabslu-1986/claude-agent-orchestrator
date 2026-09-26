"""Per-session conversation memory.

In-memory dict, intentionally the simplest thing that could work. The
interface (`get`, `append`, `clear`) is what the orchestrator depends on, so
swapping this for Redis or a database table in production is a one-file
change and does not touch routing, agent, or tool logic.

LGPD note (Etapa 11): conversation history is personal data under Art. 5,
I -- keeping it forever by default is exactly the kind of unbounded
retention Art. 6, III (necessidade) argues against. `ttl_hours` makes
retention an explicit, timestamped decision instead of "however long the
process happens to stay up": every message is stamped when it's stored, and
a session's messages older than the TTL are dropped the next time that
session is touched. `clear()` (unchanged) is this module's half of the
right to erasure -- see Orchestrator.forget_session for the other half
(there is currently nothing else to erase, since the knowledge base holds
no per-customer data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=_utcnow)


class SessionMemory:
    def __init__(self, ttl_hours: float | None = None):
        """ttl_hours: if set, messages older than this are dropped whenever
        their session is read or appended to. None (the default) keeps
        today's behavior -- no automatic expiry.
        """
        self._sessions: dict[str, list[Message]] = {}
        self._ttl = timedelta(hours=ttl_hours) if ttl_hours is not None else None

    def _prune(self, session_id: str) -> None:
        if self._ttl is None or session_id not in self._sessions:
            return
        cutoff = _utcnow() - self._ttl
        kept = [m for m in self._sessions[session_id] if m.timestamp >= cutoff]
        if kept:
            self._sessions[session_id] = kept
        else:
            self._sessions.pop(session_id, None)

    def get(self, session_id: str) -> list[Message]:
        self._prune(session_id)
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, role: str, content: str) -> None:
        self._prune(session_id)
        self._sessions.setdefault(session_id, []).append(Message(role=role, content=content))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def as_api_messages(self, session_id: str) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.get(session_id)]
