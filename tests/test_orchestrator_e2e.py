"""Scenario 8-10: full pipeline — router -> specialist -> tools -> memory,
covering the three outcomes a real deployment has to handle: a grounded
answer, a tool-driven answer, and an escalation.
"""

from pathlib import Path

from conftest import FakeClaudeClient, text_response, tool_use_response

from orchestrator.orchestrator import Orchestrator

KB_DIR = Path(__file__).parent.parent / "src" / "orchestrator" / "knowledge_base"


def _build_orchestrator(scripted_responses):
    client = FakeClaudeClient(scripted_responses)
    return Orchestrator(client=client, knowledge_base_dir=KB_DIR)


def test_e2e_billing_question_answered_via_knowledge_base_tool():
    orch = _build_orchestrator(
        [
            text_response("billing"),  # router
            tool_use_response(  # specialist decides to search the KB
                "search_knowledge_base", {"query": "refund timeline after cancellation"}
            ),
            text_response(  # specialist's final answer after seeing the tool result
                "Refunds are prorated to your cancellation date and land back on your "
                "card within 5 to 10 business days."
            ),
        ]
    )

    reply = orch.handle_message("session-1", "If I cancel today, when do I get refunded?")

    assert reply.intent == "billing"
    assert reply.agent_name == "billing"
    assert reply.escalated is False
    assert "5 to 10 business days" in reply.text
    assert reply.tool_calls_made == ["search_knowledge_base"]


def test_e2e_sales_question_answered_directly_without_tools():
    orch = _build_orchestrator(
        [
            text_response("sales"),  # router
            text_response("Sure — could you tell me roughly how many seats you need?"),
        ]
    )

    reply = orch.handle_message("session-2", "I'm interested in your Growth plan")

    assert reply.intent == "sales"
    assert reply.escalated is False
    assert reply.tool_calls_made == []


def test_e2e_agent_escalates_when_it_cannot_help():
    orch = _build_orchestrator(
        [
            text_response("billing"),  # router
            tool_use_response(  # specialist gives up and escalates
                "escalate_to_human",
                {"reason": "customer disputes a charge and wants a manual chargeback review"},
            ),
        ]
    )

    reply = orch.handle_message("session-3", "I want to dispute a charge from 3 months ago")

    assert reply.escalated is True
    assert "chargeback" in reply.escalation_reason


def test_e2e_router_unknown_intent_escalates_without_calling_a_specialist():
    orch = _build_orchestrator([text_response("not-a-real-intent")])  # router only

    reply = orch.handle_message("session-4", "asdkjfh unrelated gibberish")

    assert reply.intent == "unknown"
    assert reply.agent_name == "router"
    assert reply.escalated is True


def test_e2e_conversation_history_is_passed_to_the_specialist_on_second_turn():
    client = FakeClaudeClient(
        [
            text_response("support"),  # router, turn 1
            text_response("Sure, what's the order number?"),  # specialist, turn 1
            text_response("support"),  # router, turn 2
            text_response("Thanks, checking that now."),  # specialist, turn 2
        ]
    )
    orch = Orchestrator(client=client, knowledge_base_dir=KB_DIR)

    orch.handle_message("session-5", "My package hasn't arrived")
    orch.handle_message("session-5", "The order number is 4471")

    # the second specialist call must include the full prior turn, not just
    # the newest message
    second_specialist_call = client.calls[3]
    roles_and_content = [(m["role"], m["content"]) for m in second_specialist_call["messages"]]
    assert ("user", "My package hasn't arrived") in roles_and_content
    assert ("assistant", "Sure, what's the order number?") in roles_and_content
    assert ("user", "The order number is 4471") in roles_and_content


def test_e2e_tool_loop_gives_up_gracefully_after_max_iterations():
    # Specialist keeps calling search_knowledge_base and never converges to a
    # final text answer — the loop must bail out to escalation rather than
    # hang or crash.
    responses = [text_response("support")] + [
        tool_use_response("search_knowledge_base", {"query": "irrelevant"}, call_id=f"c{i}")
        for i in range(5)
    ]
    orch = _build_orchestrator(responses)

    reply = orch.handle_message("session-6", "some open-ended question")

    assert reply.escalated is True
    assert "max iterations" in reply.escalation_reason
