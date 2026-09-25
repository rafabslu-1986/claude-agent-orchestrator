"""Per-session conversation memory.

In-memory dict, intentionally the simplest thing that could work. The
interface (`get`, `append`, `clear`) is what the orchestrator depends on, so
swapping this for Redis or a database table in production is a one-file
change and does not touch routing, agent, or tool logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


class SessionMemory:
    def __init__(self):
        self._sessions: dict[str, list[Message]] = {}

    def get(self, session_id: str) -> list[Message]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, role: str, content: str) -> None:
        self._sessions.setdefault(session_id, []).append(Message(role=role, content=content))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def as_api_messages(self, session_id: str) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.get(session_id)]
