"""Retrieval eval scenarios for the knowledge base - a labeled query set.

Two kinds of scenario:

- A "positive" scenario (expected_source set) pairs a realistic customer
  question with the knowledge base file that should come back in the
  results -- used to measure Hit Rate@k and MRR.
- A "negative" scenario (expected_source=None) is a realistic customer
  question the knowledge base genuinely cannot answer. The correct system
  behavior is to retrieve nothing confident enough to ground an answer on,
  so the specialist agent escalates instead of the model improvising an
  answer from a loosely related paragraph. This is the more important half
  of the eval: a RAG layer that always finds *something* is a hallucination
  risk, not a feature.

Used by evals/rag_eval.py. Needs no API key and no network access -- it only
exercises KnowledgeBase.search.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RagScenario:
    id: str
    query: str
    expected_source: str | None  # None = should NOT retrieve anything confident


RAG_SCENARIOS: list[RagScenario] = [
    # --- positive: answerable from the knowledge base ---
    RagScenario("refund_timing", "how long does a refund take after cancellation", "billing_faq.md"),
    RagScenario("invoice_when", "when are invoices generated each month", "billing_faq.md"),
    RagScenario("payment_method_update", "who is allowed to update the card on file", "billing_faq.md"),
    RagScenario("failed_payment", "what happens after a payment fails three times", "billing_faq.md"),
    RagScenario("dispute_charge", "customer wants to dispute a charge through their bank", "billing_faq.md"),
    RagScenario("plan_difference", "what does the Growth plan include", "product_plans.md"),
    RagScenario("enterprise_sso", "does the Enterprise plan include single sign-on", "product_plans.md"),
    RagScenario("upgrade_proration", "does upgrading a plan get prorated right away", "product_plans.md"),
    RagScenario("free_trial_length", "is there a free trial and does it need a payment method", "product_plans.md"),
    RagScenario("downgrade_timing", "when does a plan downgrade take effect", "product_plans.md"),
    RagScenario("shipping_time", "how many business days does standard shipping take", "shipping_policy.md"),
    RagScenario("international_customs", "who pays customs duties on an international order", "shipping_policy.md"),
    RagScenario("tracking_number", "when does the customer get a tracking number after shipping", "shipping_policy.md"),
    RagScenario("redirect_address", "can a shipped order be redirected to a new address", "shipping_policy.md"),
    RagScenario("lost_package", "package lost in transit, refund or replacement", "shipping_policy.md"),
    # --- negative: sounds on-topic, but the knowledge base has no answer ---
    RagScenario("student_discount", "do you offer a discount for students or nonprofits", None),
    RagScenario("mobile_app", "is there a mobile app to track my order", None),
    RagScenario("accepted_currencies", "what currencies do you accept for payment", None),
    RagScenario("po_box_shipping", "do you ship to PO boxes", None),
    RagScenario("pause_subscription", "can I pause my subscription instead of canceling it", None),
    RagScenario("free_shipping_threshold", "what order total qualifies for free shipping", None),
]
