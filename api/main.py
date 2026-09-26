"""Etapa 6: HTTP entry point for a real deployment.

Wraps `Orchestrator.handle_message` as a small FastAPI app so the framework
can sit behind a public URL instead of only running through pytest or
`examples/demo_conversation.py`. Nothing about the orchestration itself
(routing, tools, memory, budget, resilience) changes here -- this file adds
an HTTP shape around the exact same `Orchestrator` class the test suite
already exercises, in ~60 lines.

Configuration is via environment variables, read once at import time:

    ANTHROPIC_API_KEY                  required for live calls (claude_client.py)
    HELICONE_API_KEY                   optional cost/latency observability (Etapa 8)
    ORCHESTRATOR_RESILIENT             "false" to disable retry + circuit breaker
                                        (Etapa 13); defaults to enabled in this
                                        entry point, unlike the library default
    ORCHESTRATOR_MEMORY_TTL_HOURS      optional float, LGPD memory TTL (Etapa 11)
    ORCHESTRATOR_MAX_SESSION_COST_USD  optional float, per-session cost cap (Etapa 14)

Run locally:
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn api.main:app --reload

Interactive docs (and a way to try it by hand) at /docs once running.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from orchestrator.orchestrator import Orchestrator  # noqa: E402


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes")


app = FastAPI(
    title="Claude Agent Orchestrator API",
    description=(
        "HTTP wrapper around the multi-agent orchestrator: routes an inbound "
        "message to a specialist (sales/support/billing), grounds policy "
        "answers in the local knowledge base, and escalates to a human when "
        "needed. See the project README for the full architecture."
    ),
    version="1.0.0",
)

# Built once at process startup, reused across every request -- this is the
# whole reason Etapa 6 is a persistent process (Railway) rather than a
# stateless serverless function (Vercel): SessionMemory and SessionBudget
# are plain in-process dicts, and a conversation only stays coherent across
# turns if the same process (and the same dict) answers every request for
# that session_id. See "Por que Railway em vez de Vercel" in the README.
_orchestrator = Orchestrator(
    resilient=_env_bool("ORCHESTRATOR_RESILIENT", default=True),
    memory_ttl_hours=_env_float("ORCHESTRATOR_MEMORY_TTL_HOURS"),
    max_session_cost_usd=_env_float("ORCHESTRATOR_MAX_SESSION_COST_USD"),
)


class MessageIn(BaseModel):
    session_id: str
    message: str


class MessageOut(BaseModel):
    session_id: str
    intent: str
    agent_name: str
    text: str
    escalated: bool
    escalation_reason: str | None
    tool_calls_made: list[str]


class ExportedMessage(BaseModel):
    role: str
    content: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/messages", response_model=MessageOut)
def post_message(payload: MessageIn) -> MessageOut:
    if not payload.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    reply = _orchestrator.handle_message(payload.session_id, payload.message)
    return MessageOut(
        session_id=reply.session_id,
        intent=reply.intent,
        agent_name=reply.agent_name,
        text=reply.text,
        escalated=reply.escalated,
        escalation_reason=reply.escalation_reason,
        tool_calls_made=reply.tool_calls_made,
    )


@app.get("/sessions/{session_id}", response_model=list[ExportedMessage])
def export_session(session_id: str) -> list[dict[str, str]]:
    """LGPD Art. 18, II/V -- direito de acesso e portabilidade (Etapa 11)."""
    return _orchestrator.export_session_data(session_id)


@app.delete("/sessions/{session_id}", status_code=204)
def forget_session(session_id: str) -> None:
    """LGPD Art. 18, VI -- direito a eliminacao (Etapa 11)."""
    _orchestrator.forget_session(session_id)
