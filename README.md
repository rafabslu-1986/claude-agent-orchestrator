# Claude Agent Orchestrator

A multi-agent orchestration framework built directly on the Claude API — no no-code layer in between. It routes an inbound message to the right specialist, grounds every policy-level answer in a real knowledge base through native tool use, and hands off to a human the moment an agent isn't confident.

This generalizes the routing/handoff pattern from a production WhatsApp + Instagram customer service system I built and run for a travel agency, rewritten here as a reusable, client-agnostic framework so the architecture itself — not one company's data — is what's on display.

## What it does

1. **Routes** an inbound message to one of three specialists (`sales`, `support`, `billing`) using a single, cheap Claude classification call. Anything it can't confidently classify goes straight to `unknown` instead of being guessed at.
2. **Answers** using a specialist agent configured with the **FPCL prompt framework** — Role, Context, Limits, Format — so every agent's behavior is defined by four short, auditable sections instead of one long free-form prompt.
3. **Grounds** policy answers (prices, refund timelines, shipping windows) in a local knowledge base via Claude's native tool use, instead of letting the model answer from assumptions.
4. **Escalates** to a human, with a logged reason, whenever a specialist calls the `escalate_to_human` tool — the same fallback the production system uses on the human-handoff step.
5. **Remembers** the conversation across turns per session, so a specialist sees the full thread, not just the latest message.

## Architecture

```
inbound message
      │
      ▼
   Router (1 Claude call, no tools)
      │
      ├── sales / support / billing ──▶ Specialist agent
      │                                     │
      │                        ┌────────────┴────────────┐
      │                        ▼                          ▼
      │              search_knowledge_base        escalate_to_human
      │               (TF-IDF over local docs)      (flags for a human)
      │                        │
      │                        ▼
      │                 grounded final answer
      │
      └── unknown ──▶ escalate_to_human directly
```

Every arrow above is a real function call in this repo, not a diagram aspiration — see `src/orchestrator/orchestrator.py` for the whole pipeline in about 40 lines.

## Why TF-IDF instead of an embeddings API

The retrieval layer (`rag.py`) uses TF-IDF + cosine similarity over local markdown files, not an embeddings API. That's a scope decision, not an architectural limitation: `KnowledgeBase.search(query, top_k)` is the only interface the rest of the code depends on, so swapping in a real vector database (pgvector, Pinecone, etc.) with embeddings is a one-file change — nothing in the router, agents, or orchestrator needs to know. Keeping it TF-IDF means the whole project installs, runs, and tests fully offline with no external API cost.

## Key results

- **16 automated test scenarios, all passing** — routing (including the unknown-intent fallback), RAG retrieval accuracy, both tools, session memory isolation, and 6 full end-to-end pipeline scenarios (grounded answer, direct answer, escalation, multi-turn memory, and the tool-loop giving up gracefully after too many iterations).
- Tests run **fully offline** against a scripted fake Claude client (`tests/conftest.py`) — no API key or network access needed to verify the logic. A separate `examples/demo_conversation.py` script is provided for a live run against the real API.
- The tool-use loop has a hard iteration cap with a graceful escalation fallback, so a specialist that can't converge on an answer degrades to a human handoff instead of hanging or erroring.

## Project layout

```
src/orchestrator/
  claude_client.py   thin wrapper around the Anthropic SDK (the only file that imports it)
  prompts.py          FPCL system-prompt builder
  router.py           intent classification
  agents.py           specialist agents + the shared tool-use loop
  rag.py               TF-IDF knowledge base retrieval
  tools.py             tool schemas + local tool implementations
  memory.py            per-session conversation state
  orchestrator.py     ties it all together
  knowledge_base/      sample markdown docs the specialists search against
tests/                16 scenarios, run offline against a fake client
examples/              live demo script (needs a real API key)
```

## Installation

```bash
git clone https://github.com/rafabslu-1986/claude-agent-orchestrator
cd claude-agent-orchestrator
pip install -r requirements.txt
```

## Running the tests

```bash
pytest tests/ -v
```

No API key needed — every test runs against a scripted fake Claude client.

## Running the live demo

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/demo_conversation.py
```

## Extending it

- **New specialist**: write one new `FPCLPrompt` in `agents.py`, add it to `Orchestrator.specialists`, add its name to `router.INTENTS`.
- **New tool**: add a schema to `TOOL_SCHEMAS` in `tools.py` and a handler method on `ToolExecutor`.
- **Real vector DB**: implement the same `search(query, top_k)` interface as `KnowledgeBase` and swap it in — no other file changes.
- **Persistent memory**: swap the dict in `SessionMemory` for Redis or a database table behind the same `get` / `append` / `clear` interface.

## License

MIT
