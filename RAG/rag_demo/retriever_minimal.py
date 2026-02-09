"""Lightweight cosine-similarity retriever using only the standard library."""

from __future__ import annotations

import math

from embeddings import embed


class MinimalRetriever:
    """Retrieve the most similar document using pure-Python cosine similarity."""

    def __init__(self, data: list[str]) -> None:
        self.data = data
        self.embeddings = [embed(d) for d in data]

    def retrieve(self, query: str, top_k: int = 1) -> str:
        """Return the single best-matching document for *query*."""
        q_emb = embed(query)
        sims = [_cosine(q_emb, e) for e in self.embeddings]
        return self.data[sims.index(max(sims))]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-8)
