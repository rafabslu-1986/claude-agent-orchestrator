"""Multi-agent orchestration framework built directly on the Claude API.

Core pieces:
- claude_client: thin wrapper around the Anthropic SDK (mockable for tests)
- prompts: the Role / Context / Limits / Format (FPCL) system-prompt builder
- router: intent classification that decides which specialist handles a message
- agents: specialist agents (sales, support, billing), each with its own FPCL prompt
- rag: BM25 based retrieval over a local knowledge base
- tools: Claude native tool-use definitions and their local implementations
- memory: per-session conversation state, with optional LGPD-driven TTL
- privacy: PII detection used by the knowledge-base guardrail test
- resilience: optional retry-with-backoff + circuit breaker around any
  .send()-shaped client, for transient Claude API failures
- budget: optional per-session cost cap, fed by real token usage, that
  escalates to a human once a session crosses it
- orchestrator: ties everything together, including human handoff, the
  LGPD data-subject rights (export_session_data, forget_session), optional
  resilience (resilient=True) and the optional cost cap
  (max_session_cost_usd)
"""

__version__ = "0.1.0"
