"""Eval scenarios for the router + specialists - 20 real-shaped conversations.

Each scenario is a single user message plus the expected routing outcome.
Used by evals/run_evals.py to measure the orchestrator's intent-routing
accuracy against the real Claude API.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Scenario:
    id: str
    message: str
    expected_intent: str
    expected_escalated: bool | None = None


SCENARIOS: list[Scenario] = [
    Scenario("sales_pricing", "How much does the Growth plan cost per month?", "sales"),
    Scenario("sales_compare", "What is the difference between Starter and Growth?", "sales"),
    Scenario("sales_trial", "Can I try this out before paying for anything?", "sales"),
    Scenario("sales_upgrade", "I am on Starter now, how do I move to Enterprise?", "sales"),
    Scenario("support_howto", "How do I connect my Slack workspace to this?", "support"),
    Scenario("support_bug", "The dashboard has been showing a blank screen since this morning.", "support"),
    Scenario("support_feature", "Does this integrate with Zapier?", "support"),
    Scenario("support_error", "I keep getting a 500 error when I try to export my data.", "support"),
    Scenario("billing_refund", "I cancelled yesterday, when do I get my money back?", "billing"),
    Scenario("billing_invoice", "Can you send me the invoice for last month?", "billing"),
    Scenario("billing_payment", "I need to update the card on file, it expired.", "billing"),
    Scenario("billing_cancel", "I want to cancel my subscription entirely.", "billing"),
    Scenario("unknown_weather", "What is the weather like in Lisbon today?", "unknown", expected_escalated=True),
    Scenario("unknown_chitchat", "lol did you see that game last night", "unknown", expected_escalated=True),
    Scenario("unknown_unrelated", "Can you recommend a good pizza place nearby?", "unknown", expected_escalated=True),
    Scenario("unknown_vague", "I have a question.", "unknown", expected_escalated=True),
    Scenario("edge_mixed_intent", "I want to cancel my subscription because your support never answered my ticket.", "billing"),
    Scenario("edge_short", "refund?", "billing"),
    Scenario("edge_portuguese", "Oi, quero saber quanto custa o plano Growth por mes.", "sales"),
    Scenario("edge_frustrated", "This is the third time I am writing and nobody fixed my billing issue", "billing"),
]
