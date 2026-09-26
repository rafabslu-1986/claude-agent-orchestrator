"""Run every scenario in evals/scenarios.py against the REAL Claude API and
report a pass rate.

This is an outcome-based eval: instead of testing individual functions in
isolation, it drives the orchestrator exactly like a real user message would,
and checks whether the router sent it to the right specialist (and whether
it escalated when it should have).

Usage:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    python evals/run_evals.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.orchestrator import Orchestrator  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


def run() -> None:
    orchestrator = Orchestrator()
    results = []
    passed = 0

    for scenario in SCENARIOS:
        session_id = f"eval-{scenario.id}"
        reply = orchestrator.handle_message(session_id, scenario.message)

        intent_ok = reply.intent == scenario.expected_intent
        escalated_ok = (
            scenario.expected_escalated is None
            or reply.escalated == scenario.expected_escalated
        )
        ok = intent_ok and escalated_ok

        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {scenario.id}: "
            f"expected={scenario.expected_intent} got={reply.intent} "
            f"escalated={reply.escalated}"
        )

        results.append(
            {
                "id": scenario.id,
                "message": scenario.message,
                "expected_intent": scenario.expected_intent,
                "got_intent": reply.intent,
                "expected_escalated": scenario.expected_escalated,
                "got_escalated": reply.escalated,
                "pass": ok,
            }
        )

    total = len(SCENARIOS)
    print(f"\n{passed}/{total} passed ({passed / total:.0%})")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval-{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    run()
