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


def test_min_token_overlap_rejects_single_keyword_coincidence(knowledge_base_dir):
    """A query that only shares one distinctive word with a chunk can still
    score well under BM25 (or TF-IDF cosine) even when the chunk doesn't
    actually answer the question -- see evals/rag_eval.py for the measured
    false-retrieval rate this causes. min_token_overlap is the opt-in gate
    that catches this case.
    """
    kb = KnowledgeBase(knowledge_base_dir)

    without_gate = kb.search("do you offer a discount for students or nonprofits", top_k=3)
    assert without_gate, "expected the ungated search to still find a lexical match"

    with_gate = kb.search(
        "do you offer a discount for students or nonprofits", top_k=3, min_token_overlap=2
    )
    assert with_gate == []


def test_min_token_overlap_keeps_genuine_multi_keyword_matches(knowledge_base_dir):
    kb = KnowledgeBase(knowledge_base_dir)

    results = kb.search(
        "package lost in transit, refund or replacement", top_k=3, min_token_overlap=2
    )

    assert results
    assert any("shipping_policy.md" == r.source for r in results)
