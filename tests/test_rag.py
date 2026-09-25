"""Scenario 3-4: RAG retrieval against the real knowledge base on disk."""

from orchestrator.rag import KnowledgeBase


def test_knowledge_base_loads_all_documents(knowledge_base_dir):
    kb = KnowledgeBase(knowledge_base_dir)
    assert kb.size > 0


def test_search_returns_relevant_chunk_for_refund_query(knowledge_base_dir):
    kb = KnowledgeBase(knowledge_base_dir)

    results = kb.search("how long does a refund take after cancellation", top_k=3)

    assert results, "expected at least one matching chunk"
    assert any("billing_faq.md" == r.source for r in results)
    assert any("prorated" in r.text.lower() for r in results)


def test_search_returns_nothing_for_unrelated_query(knowledge_base_dir):
    kb = KnowledgeBase(knowledge_base_dir)

    results = kb.search("what is the capital of France", top_k=3, min_score=0.2)

    assert results == []
