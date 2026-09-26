"""Top-level entry point: router -> specialist -> (tools + memory) -> reply.

This is the piece a real deployment (WhatsApp webhook, web chat widget,
Instagram DM handler) calls per inbound message. Everything else in this
package exists to make this function's job simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agents import Specialist, build_billing_agent, build_sales_agent, build_support_agent
from .budget import SessionBudget
from .claude_client import ClaudeClient
from .memory import SessionMemory
from .rag import KnowledgeBase
from .resilience import CircuitBreaker, ResilientClaudeClient, RetryConfig
from .router import Router
from .tools import ToolExecutor

DEFAULT_KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"


@dataclass
class OrchestratorReply:
    session_id: str
    intent: str
    agent_name: str
    text: str
    escalated: bool
    escalation_reason: str | None
    tool_calls_made: list[str]


class Orchestrator:
    def __init__(
        self,
        client: ClaudeClient | None = None,
        knowledge_base_dir: str | Path = DEFAULT_KNOWLEDGE_BASE_DIR,
        memory_ttl_hours: float | None = None,
        resilient: bool = False,
        retry_config: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        max_session_cost_usd: float | None = None,
        budget: SessionBudget | None = None,
    ):
        base_client = client or ClaudeClient()
        # Etapa 13: opt-in, off by default -- today's behavior (an
        # unhandled exception from a transient API failure bubbles straight
        # up) is unchanged unless resilient=True. ResilientClaudeClient is a
        # pure wrapper around whatever exposes .send(...), so router,
        # agents and tools never need to know it exists.
        self.client = (
            ResilientClaudeClient(
                base_client, retry_config=retry_config, circuit_breaker=circuit_breaker
            )
            if resilient
            else base_client
        )
        self.memory = SessionMemory(ttl_hours=memory_ttl_hours)
        # Etapa 14: opt-in, off by default -- a session never gets cut off
        # unless max_session_cost_usd (or a pre-built budget) is passed in.
        self.budget = budget or (
            SessionBudget(max_cost_usd=max_session_cost_usd)
            if max_session_cost_usd is not None
            else None
        )
        self.knowledge_base = KnowledgeBase(knowledge_base_dir)
        self.router = Router(self.client)

        tool_executor = ToolExecutor(self.knowledge_base)
        self.specialists: dict[str, Specialist] = {
            "sales": build_sales_agent(self.client, tool_executor),
            "support": build_support_agent(self.client, tool_executor),
            "billing": build_billing_agent(self.client, tool_executor),
        }

    def handle_message(self, session_id: str, user_message: str) -> OrchestratorReply:
        if self.budget is not None and self.budget.exceeded(session_id):
            # Etapa 14: fail fast, like the circuit breaker in Etapa 13 --
            # don't spend a single extra token on a session that already
            # crossed its cap. Still log the customer's message so a human
            # taking over sees the full thread.
            self.memory.append(session_id, "user", user_message)
            reply = OrchestratorReply(
                session_id=session_id,
                intent="unknown",
                agent_name="budget_guardrail",
                text=(
                    "I want to make sure this goes to the right person — connecting "
                    "you with a human teammate now."
                ),
                escalated=True,
                escalation_reason=(
                    f"session cost budget exceeded (spent ${self.budget.spent(session_id):.4f} "
                    f"of a ${self.budget.max_cost_usd:.2f} cap)"
                ),
                tool_calls_made=[],
            )
            self.memory.append(session_id, "assistant", reply.text)
            return reply

        self.memory.append(session_id, "user", user_message)
        conversation = self.memory.as_api_messages(session_id)

        route = self.router.classify(user_message)
        self._record_usage(session_id, route.usage)

        if route.intent == "unknown":
            reply = OrchestratorReply(
                session_id=session_id,
                intent=route.intent,
                agent_name="router",
                text=(
                    "I want to make sure this goes to the right person — connecting "
                    "you with a human teammate now."
                ),
                escalated=True,
                escalation_reason="router could not confidently classify the message",
                tool_calls_made=[],
            )
            self.memory.append(session_id, "assistant", reply.text)
            return reply

        specialist = self.specialists[route.intent]
        result = specialist.handle(conversation)
        self._record_usage(session_id, result.usage)

        self.memory.append(session_id, "assistant", result.text)

        return OrchestratorReply(
            session_id=session_id,
            intent=route.intent,
            agent_name=result.agent_name,
            text=result.text,
            escalated=result.escalated,
            escalation_reason=result.escalation_reason,
            tool_calls_made=result.tool_calls_made,
        )

    def _record_usage(self, session_id: str, usage: dict[str, int] | None) -> None:
        if self.budget is not None and usage is not None:
            self.budget.record(session_id, usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    def export_session_data(self, session_id: str) -> list[dict[str, str]]:
        """LGPD Art. 18, II and V: direito de acesso and portabilidade dos
        dados. Everything this system holds about one session/customer, in
        a plain, exportable shape -- the same messages the specialist sees,
        nothing hidden behind it, because there is nothing else: this
        orchestrator has no other per-session store.
        """
        return self.memory.as_api_messages(session_id)

    def forget_session(self, session_id: str) -> None:
        """LGPD Art. 18, VI: direito a eliminacao (right to erasure). Wipes
        this session's conversation history. In a deployment where
        SessionMemory has been swapped for Redis/a DB (see memory.py), this
        is the one call site that needs to also touch whatever other
        per-customer stores were added alongside it.
        """
        self.memory.clear(session_id)
