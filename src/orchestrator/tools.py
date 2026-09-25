"""Claude native tool-use: schemas plus the local functions that execute them.

Two tools are defined:

- search_knowledge_base: lets any specialist agent pull grounded context from
  the RAG index instead of answering from the model's own assumptions.
- escalate_to_human: lets an agent end its own turn and flag the session for
  a human, with a reason. This is the same "human handoff when the agent
  isn't confident" pattern used in the WhatsApp/Instagram production system
  this framework generalizes from.

Tool schemas follow the Anthropic Messages API tool-use format directly, so
they can be passed to ClaudeClient.send(tools=...) unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .rag import KnowledgeBase

SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
ESCALATE_TO_HUMAN = "escalate_to_human"


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": SEARCH_KNOWLEDGE_BASE,
        "description": (
            "Search the company knowledge base for policy or product information "
            "relevant to the customer's question. Always use this before stating "
            "a policy detail (prices, timeframes, refund rules) instead of guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query, not the raw customer message.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": ESCALATE_TO_HUMAN,
        "description": (
            "End the conversation turn and flag it for a human agent. Use this when "
            "the customer explicitly asks for a human, when the request involves "
            "something you have no tool or knowledge base access to (e.g. a specific "
            "account action), or when you are not confident in the answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Short reason for the handoff, for the human agent's queue.",
                }
            },
            "required": ["reason"],
        },
    },
]


@dataclass
class ToolResult:
    tool_use_id: str
    content: str
    escalated: bool = False
    escalation_reason: str | None = None


class ToolExecutor:
    """Executes tool calls returned by Claude against real local implementations."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self._knowledge_base = knowledge_base
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            SEARCH_KNOWLEDGE_BASE: self._search_knowledge_base,
            ESCALATE_TO_HUMAN: self._escalate_to_human,
        }

    def execute(self, tool_call: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(tool_call["name"])
        if handler is None:
            return ToolResult(
                tool_use_id=tool_call["id"],
                content=f"Unknown tool: {tool_call['name']}",
            )
        return handler(tool_call)

    def _search_knowledge_base(self, tool_call: dict[str, Any]) -> ToolResult:
        query = tool_call["input"]["query"]
        chunks = self._knowledge_base.search(query, top_k=3)
        if not chunks:
            content = "No relevant knowledge base entries found for this query."
        else:
            content = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)
        return ToolResult(tool_use_id=tool_call["id"], content=content)

    def _escalate_to_human(self, tool_call: dict[str, Any]) -> ToolResult:
        reason = tool_call["input"]["reason"]
        return ToolResult(
            tool_use_id=tool_call["id"],
            content="Escalation logged. A human agent will take over this conversation.",
            escalated=True,
            escalation_reason=reason,
        )
