"""Retrieval over a local knowledge base.

Ranking is BM25 (Okapi BM25, via the tiny pure-Python `rank_bm25` package),
not raw TF-IDF + cosine similarity, and not an embeddings API. BM25 adds two
things plain TF-IDF cosine similarity does not: term-frequency saturation (a
chunk that repeats a query word ten times doesn't score ten times higher)
and per-document length normalization. It is also lighter: this drops
scikit-learn and numpy from the dependency list entirely.

That said, `evals/rag_eval.py` shows the honest limit of this (or any
purely lexical) ranking: on a small knowledge base, a query that shares even
one distinctive word with a chunk ("Do you offer a student discount?" vs. a
chunk about "the Growth plan discount for annual billing") can score high
enough to be retrieved with confidence, even though the chunk doesn't
actually answer the question. Neither switching TF-IDF for BM25 nor raising
the score threshold cleanly fixes this without also losing genuine matches
-- see the eval output for the measured trade-off. The mitigation shipped
here is `min_token_overlap`: an opt-in, interpretable second gate (require
at least N distinctive words in common, not just a high score) that a
caller can turn on when it is willing to trade recall for precision. It is
off by default so existing callers keep today's (permissive) behavior.

Scope choice, same as before: the KnowledgeBase interface
(`search(query, top_k) -> list[Chunk]`) is the only thing the rest of the
codebase depends on, so swapping in a real vector database with embeddings
later is still a one-file change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# Stripped before BM25 indexing, not just for the overlap gate below. This
# matters more than it looks: rank_bm25's negative-idf smoothing (its
# `epsilon` parameter) gives a near-universal word like "of" a small
# *positive* score instead of ~0. Left unstripped, a query that only shares
# a stopword with the corpus ("what is the capital of France") would still
# come back as a nonzero, deceptively confident match.
_STOPWORDS = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves
    he him his himself she her hers herself it its itself they them their
    theirs themselves what which who whom this that these those am is are
    was were be been being have has had having do does did doing a an the
    and but if or because as until while of at by for with about against
    between into through during before after above below to from up down in
    out on off over under again further then once here there when where why
    how all any both each few more most other some such no nor not only own
    same so than too very s t can will just don should now d ll m o re ve y
    ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn
    shan shouldn wasn weren won wouldn
    """.split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _distinctive_tokens(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) > 2}


@dataclass
class Chunk:
    source: str
    text: str
    score: float = 0.0


def _split_into_chunks(text: str, source: str) -> list[Chunk]:
    """Split a document into paragraph-sized chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Chunk(source=source, text=p) for p in paragraphs]


class KnowledgeBase:
    def __init__(self, docs_dir: str | Path):
        self.docs_dir = Path(docs_dir)
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {self.docs_dir}")

        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            self._chunks.extend(_split_into_chunks(text, source=path.name))

        if not self._chunks:
            raise ValueError(f"No .md documents found in {self.docs_dir}")

        tokenized_corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.05,
        min_token_overlap: int | None = None,
    ) -> list[Chunk]:
        """Return up to `top_k` chunks ranked by BM25 score.

        min_score: BM25's raw score scale is unbounded and corpus-dependent
        (unlike cosine similarity's [0, 1] range), so this is a weak floor
        mainly meant to drop true zero-overlap matches, not a precision
        control. Left at the historical TF-IDF-era default for continuity.

        min_token_overlap: optional, off by default. When set, a chunk must
        also share at least this many distinctive (non-stopword) tokens with
        the query, in addition to clearing min_score. This is the knob that
        actually improves precision against superficially-similar-but-wrong
        matches -- see the module docstring and evals/rag_eval.py.
        """
        assert self._bm25 is not None
        scores = self._bm25.get_scores(_tokenize(query))

        ranked = sorted(zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True)

        query_tokens = _distinctive_tokens(query) if min_token_overlap else set()

        results: list[Chunk] = []
        for chunk, score in ranked[:top_k]:
            if score < min_score:
                continue
            if min_token_overlap and len(query_tokens & _distinctive_tokens(chunk.text)) < min_token_overlap:
                continue
            results.append(Chunk(source=chunk.source, text=chunk.text, score=float(score)))
        return results

    @property
    def size(self) -> int:
        return len(self._chunks)
