"""Live demo: run a short conversation through the orchestrator.

Requires a real ANTHROPIC_API_KEY in the environment — this script makes
actual API calls and is not part of the automated test suite (see tests/,
which run fully offline against a fake client).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/demo_conversation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.orchestrator import Orchestrator


def main() -> None:
    orchestrator = Orchestrator()
    session_id = "demo-session"

    turns = [
        "Hi, if I cancel my subscription today, when do I actually get my money back?",
        "And what's the difference between the Growth and Enterprise plans?",
        "This is a totally unrelated question about the weather on Mars.",
    ]

    for user_message in turns:
        print(f"\nCustomer: {user_message}")
        reply = orchestrator.handle_message(session_id, user_message)
        print(f"[{reply.intent} -> {reply.agent_name}] {reply.text}")
        if reply.tool_calls_made:
            print(f"  (tools used: {', '.join(reply.tool_calls_made)})")
        if reply.escalated:
            print(f"  (escalated to human: {reply.escalation_reason})")


if __name__ == "__main__":
    main()
