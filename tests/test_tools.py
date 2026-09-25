"""Scenario 5-6: tool execution — knowledge base search and human escalation."""

from orchestrator.rag import KnowledgeBase
from orchestrator.tools import ESCALATE_TO_HUMAN, SEARCH_KNOWLEDGE_BASE, ToolExecutor


def test_search_knowledge_base_tool_returns_grounded_content(knowledge_base_dir):
    executor = ToolExecutor(KnowledgeBase(knowledge_base_dir))

    result = executor.execute(
        {"id": "t1", "name": SEARCH_KNOWLEDGE_BASE, "input": {"query": "express shipping time"}}
    )

    assert result.escalated is False
    assert "shipping_policy.md" in result.content


def test_escalate_to_human_tool_sets_escalation_flag(knowledge_base_dir):
    executor = ToolExecutor(KnowledgeBase(knowledge_base_dir))

    result = executor.execute(
        {
            "id": "t2",
            "name": ESCALATE_TO_HUMAN,
            "input": {"reason": "customer wants to cancel their enterprise contract"},
        }
    )

    assert result.escalated is True
    assert result.escalation_reason == "customer wants to cancel their enterprise contract"
