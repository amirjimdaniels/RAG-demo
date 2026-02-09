"""NumPy-accelerated cosine-similarity retriever."""

from __future__ import annotations

import numpy as np

from embeddings import EMBEDDINGS, embed


class NumpyRetriever:
    """Retrieve the most similar document using vectorized NumPy operations."""

    def __init__(self, data: list[str]) -> None:
        self.data = data
        self.embeddings: np.ndarray = np.array([embed(d) for d in data])

    def retrieve(self, query: str, top_k: int = 1) -> str:
        """Return the single best-matching document for *query*."""
        q_emb = np.array(embed(query))
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        sims = self.embeddings @ q_emb / norms
        return self.data[int(np.argmax(sims))]
