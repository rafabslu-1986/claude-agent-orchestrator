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
- orchestrator: ties everything together, including human handoff and the
  LGPD data-subject rights (export_session_data, forget_session)
"""

__version__ = "0.1.0"
