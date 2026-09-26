"""Etapa 14: orcamento de custo por sessao (SessionBudget + estimate_cost_usd)."""

from __future__ import annotations

from orchestrator.budget import SessionBudget, estimate_cost_usd


def test_estimate_cost_uses_the_known_model_pricing():
    # 1,000,000 input tokens @ $3.00 + 1,000,000 output tokens @ $15.00
    cost = estimate_cost_usd("claude-sonnet-4-5", 1_000_000, 1_000_000)
    assert cost == 18.00


def test_estimate_cost_falls_back_to_default_pricing_for_an_unknown_model():
    known = estimate_cost_usd("claude-sonnet-4-5", 100_000, 10_000)
    unknown = estimate_cost_usd("some-future-model-not-in-the-table", 100_000, 10_000)
    assert unknown == known  # same fallback pricing, better than no guardrail at all


def test_budget_accumulates_across_multiple_calls_for_the_same_session():
    budget = SessionBudget(max_cost_usd=1.0)

    first_total = budget.record("session-a", input_tokens=10_000, output_tokens=1_000)
    second_total = budget.record("session-a", input_tokens=10_000, output_tokens=1_000)

    assert second_total > first_total
    assert budget.spent("session-a") == second_total


def test_budget_keeps_sessions_isolated():
    budget = SessionBudget(max_cost_usd=1.0)

    budget.record("session-a", input_tokens=50_000, output_tokens=5_000)

    assert budget.spent("session-a") > 0
    assert budget.spent("session-b") == 0.0


def test_budget_never_exceeds_when_the_guardrail_is_off():
    budget = SessionBudget(max_cost_usd=None)

    budget.record("session-a", input_tokens=10_000_000, output_tokens=10_000_000)

    assert budget.exceeded("session-a") is False


def test_budget_flips_to_exceeded_once_the_cap_is_crossed():
    budget = SessionBudget(max_cost_usd=0.01)

    assert budget.exceeded("session-a") is False  # nothing spent yet

    budget.record("session-a", input_tokens=10_000, output_tokens=10_000)  # well past $0.01

    assert budget.exceeded("session-a") is True


def test_reset_clears_a_sessions_spend():
    budget = SessionBudget(max_cost_usd=0.01)
    budget.record("session-a", input_tokens=10_000, output_tokens=10_000)
    assert budget.exceeded("session-a") is True

    budget.reset("session-a")

    assert budget.spent("session-a") == 0.0
    assert budget.exceeded("session-a") is False
