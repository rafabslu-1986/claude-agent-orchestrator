"""Retrieval over a local knowledge base.

Uses TF-IDF + cosine similarity (scikit-learn) rather than an embeddings API.
That's a deliberate scope choice, not a limitation of the architecture: the
KnowledgeBase interface below (`search(query, top_k) -> list[Chunk]`) is the
only thing the rest of the codebase depends on, so swapping in a vector
database (pgvector, Pinecone, etc.) with real embeddings later is a matter of
implementing the same interface — no changes needed anywhere else. TF-IDF
keeps this project runnable and testable offline, with no external API cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._load()

    def _load(self) -> None:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {self.docs_dir}")

        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            self._chunks.extend(_split_into_chunks(text, source=path.name))

        if not self._chunks:
            raise ValueError(f"No .md documents found in {self.docs_dir}")

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c.text for c in self._chunks])

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[Chunk]:
        assert self._vectorizer is not None and self._matrix is not None
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix)[0]

        ranked = sorted(
            zip(self._chunks, similarities), key=lambda pair: pair[1], reverse=True
        )
        results = [
            Chunk(source=c.source, text=c.text, score=float(score))
            for c, score in ranked[:top_k]
            if score >= min_score
        ]
        return results

    @property
    def size(self) -> int:
        return len(self._chunks)
