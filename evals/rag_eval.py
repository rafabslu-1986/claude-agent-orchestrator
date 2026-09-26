"""Evaluate retrieval quality for the KnowledgeBase, fully offline -- no
Claude API key needed, independent of the intent-routing eval in
evals/run_evals.py, which measures the LLM layer instead.

Two metrics, matching the two kinds of scenario in rag_scenarios.py:

- Hit Rate@k / MRR@k over the "positive" scenarios: did the right document
  show up in the top k results, and how high did it rank.
- False-Retrieval Rate over the "negative" scenarios: how often the
  knowledge base confidently hands back a chunk for a question it can't
  actually answer. This is the more important number in a system where a
  specialist agent only escalates to a human when search_knowledge_base
  returns nothing -- a false retrieval here becomes a fabricated policy
  answer in production, not just a missed one.

Both metrics are computed twice: once with today's default `search()` call
(the one tools.py actually uses), and once with `min_token_overlap=2`
turned on, so the precision/recall trade-off that gate buys is visible
instead of asserted.

Usage:
    python evals/rag_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.rag import KnowledgeBase  # noqa: E402
from rag_scenarios import RAG_SCENARIOS  # noqa: E402

TOP_K = 3


def _evaluate(kb: KnowledgeBase, *, min_token_overlap: int | None, verbose: bool) -> dict:
    positives = [s for s in RAG_SCENARIOS if s.expected_source is not None]
    negatives = [s for s in RAG_SCENARIOS if s.expected_source is None]

    hits = 0
    reciprocal_ranks: list[float] = []
    for scenario in positives:
        results = kb.search(
            scenario.query, top_k=TOP_K, min_score=0.0, min_token_overlap=min_token_overlap
        )
        sources = [r.source for r in results]
        if scenario.expected_source in sources:
            hits += 1
            reciprocal_ranks.append(1 / (sources.index(scenario.expected_source) + 1))
            status = "HIT "
        else:
            reciprocal_ranks.append(0.0)
            status = "MISS"
        if verbose:
            print(f"  [{status}] {scenario.id}: expected={scenario.expected_source} got={sources}")

    false_retrievals = 0
    for scenario in negatives:
        # Uses the default min_score -- the same threshold tools.py relies on
        # in production via KnowledgeBase.search(query, top_k=3).
        results = kb.search(scenario.query, top_k=TOP_K, min_token_overlap=min_token_overlap)
        if results:
            false_retrievals += 1
            status = "FALSE RETRIEVAL"
            detail = f"top={results[0].source} score={results[0].score:.3f}"
        else:
            status = "correctly declined"
            detail = ""
        if verbose:
            print(f"  [{status}] {scenario.id}: {detail}")

    return {
        "hit_rate": hits / len(positives),
        "mrr": sum(reciprocal_ranks) / len(positives),
        "false_retrieval_rate": false_retrievals / len(negatives),
        "n_pos": len(positives),
        "n_neg": len(negatives),
    }


def run() -> None:
    kb_dir = Path(__file__).parent.parent / "src" / "orchestrator" / "knowledge_base"
    kb = KnowledgeBase(kb_dir)

    print("=" * 60)
    print("Config A: default search() -- what tools.py uses today")
    print("=" * 60)
    a = _evaluate(kb, min_token_overlap=None, verbose=True)

    print()
    print("=" * 60)
    print("Config B: min_token_overlap=2 (opt-in precision gate)")
    print("=" * 60)
    b = _evaluate(kb, min_token_overlap=2, verbose=True)

    print()
    print("=" * 60)
    print("Summary  (A = default, B = min_token_overlap=2)")
    print("=" * 60)
    print(f"Hit Rate@{TOP_K}:          A={a['hit_rate']:.0%}   B={b['hit_rate']:.0%}")
    print(f"MRR@{TOP_K}:               A={a['mrr']:.3f}  B={b['mrr']:.3f}")
    print(
        f"False-Retrieval Rate:  A={a['false_retrieval_rate']:.0%}   "
        f"B={b['false_retrieval_rate']:.0%}"
    )


if __name__ == "__main__":
    run()
