"""Router agent: classifies an inbound message into an intent category.

The router is intentionally the cheapest, fastest call in the pipeline: a
single Claude turn with no tools, forced to answer with one lowercase word
from a fixed vocabulary. Anything it can't confidently classify becomes
"unknown", which the orchestrator treats as a signal to hand off to a human
rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from .claude_client import ClaudeClient
from .prompts import build_system_prompt

INTENTS = ("sales", "support", "billing", "unknown")

_ROUTER_SYSTEM_PROMPT = build_system_prompt(
    role=(
        "You are an intent classifier for a customer conversation router. "
        "You do not answer the customer; you only decide who should."
    ),
    context=(
        "Three specialist agents are available: "
        "'sales' (pricing, plans, new purchases), "
        "'support' (product usage, troubleshooting, how-to questions), "
        "'billing' (invoices, refunds, payment methods, charges)."
    ),
    limits=(
        "Never answer the customer's question yourself. "
        "If the message does not clearly fit one specialist, or the customer "
        "explicitly asks for a human, answer 'unknown' rather than guessing."
    ),
    format=(
        "Respond with exactly one lowercase word and nothing else: "
        "sales, support, billing, or unknown."
    ),
)


@dataclass
class RouteResult:
    intent: str
    raw_text: str
    usage: dict[str, int] | None = None  # Etapa 14: feeds the per-session budget guardrail


class Router:
    def __init__(self, client: ClaudeClient):
        self._client = client

    def classify(self, user_message: str) -> RouteResult:
        response = self._client.send(
            system=_ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=8,
            temperature=0.0,
        )
        intent = response.text.strip().lower()
        if intent not in INTENTS:
            intent = "unknown"
        return RouteResult(intent=intent, raw_text=response.text, usage=response.usage)
