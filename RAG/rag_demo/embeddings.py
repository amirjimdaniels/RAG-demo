"""Embedding, chunking, reranking, and retrieval utilities for the RAG demo.

``DocumentStore`` holds the current corpus, embedder, and precomputed embeddings.
It can be rebuilt with new text at any time.
"""

from __future__ import annotations

import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:  # Optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency fallback
    SentenceTransformer = None

try:  # CrossEncoder lives in the same package
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None


# ---- Embedder backends ----

class _EmbedderBase:
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class _TfidfEmbedder(_EmbedderBase):
    def __init__(self, sentences: list[str]) -> None:
        self._vectorizer = TfidfVectorizer()
        self._vectorizer.fit(sentences)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vectorizer.transform(texts).toarray()

    def encode_query(self, text: str) -> np.ndarray:
        return self._vectorizer.transform([text]).toarray()[0]


class _SentenceTransformerEmbedder(_EmbedderBase):
    # Model families that need a query-side instruction prefix
    _QUERY_PREFIXES = {
        "bge": "Represent this sentence for searching relevant passages: ",
        "e5": "query: ",
    }

    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name.lower()
        # Detect query prefix from model name or allow env override
        self._query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "")
        if not self._query_prefix:
            for family, prefix in self._QUERY_PREFIXES.items():
                if family in self._model_name:
                    self._query_prefix = prefix
                    break

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    def encode_query(self, text: str) -> np.ndarray:
        prefixed = self._query_prefix + text if self._query_prefix else text
        return self._model.encode([prefixed], normalize_embeddings=True)[0]


def _create_embedder(sentences: list[str]) -> _EmbedderBase:
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    if backend in {"st", "sentence-transformers", "sentence_transformers"}:
        try:
            return _SentenceTransformerEmbedder(model_name)
        except Exception:
            return _TfidfEmbedder(sentences)

    return _TfidfEmbedder(sentences)


# ---- Cross-encoder reranker ----

class _Reranker:
    """Cross-encoder reranker using sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers CrossEncoder is not available")
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[str],
        top_n: int = 3,
    ) -> list[tuple[str, float]]:
        """Rerank *chunks* for *query* and return the top *top_n*."""
        if not chunks:
            return []
        pairs = [[query, chunk] for chunk in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


def _create_reranker() -> _Reranker | None:
    enabled = os.getenv("RERANKER_ENABLED", "true").lower() in {"1", "true", "yes"}
    if not enabled:
        return None
    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        return _Reranker(model_name)
    except Exception:
        return None


# ---- Chunking ----

def split_into_sentences(text: str) -> list[str]:
    """Split *text* on sentence-ending punctuation."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def chunk_with_overlap(
    text: str,
    max_chunk_chars: int = 2000,
    overlap_frac: float = 0.15,
) -> list[str]:
    """Split *text* into overlapping chunks respecting sentence boundaries.

    Each chunk targets *max_chunk_chars* characters.  Adjacent chunks share
    approximately *overlap_frac* of their content for continuity.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    total = sum(len(s) for s in sentences) + len(sentences)
    if total <= max_chunk_chars:
        return [" ".join(sentences)]

    overlap_chars = int(max_chunk_chars * overlap_frac)
    chunks: list[str] = []
    start = 0

    while start < len(sentences):
        end = start
        size = 0
        while end < len(sentences) and size + len(sentences[end]) + 1 <= max_chunk_chars:
            size += len(sentences[end]) + 1
            end += 1
        if end == start:
            end = start + 1

        chunks.append(" ".join(sentences[start:end]))

        # Walk backward from end to find overlap start for next chunk
        overlap_size = 0
        overlap_start = end
        while overlap_start > start and overlap_size < overlap_chars:
            overlap_start -= 1
            overlap_size += len(sentences[overlap_start]) + 1

        start = overlap_start if overlap_start > start else end

    return chunks


# ---- Document store ----

class DocumentStore:
    """Encapsulates a corpus, its embedder, and precomputed embeddings."""

    def __init__(
        self,
        sentences: list[str] | None = None,
        *,
        raw_text: str | None = None,
        chunk_max_chars: int = 2000,
        chunk_overlap_frac: float = 0.15,
    ) -> None:
        if raw_text is not None:
            self.chunks = chunk_with_overlap(raw_text, chunk_max_chars, chunk_overlap_frac)
        elif sentences is not None:
            self.chunks = list(sentences)
        else:
            raise ValueError("Provide either 'sentences' or 'raw_text'")

        self._embedder = _create_embedder(self.chunks)
        self.embeddings: np.ndarray = self._embedder.encode(self.chunks)
        self._reranker: _Reranker | None = _create_reranker()

    # Backward-compat alias
    @property
    def sentences(self) -> list[str]:
        return self.chunks

    def embed(self, text: str) -> np.ndarray:
        """Return the embedding vector for *text* using this store's backend."""
        return self._embedder.encode_query(text)

    def retrieve_chunks(
        self,
        query: str,
        initial_k: int = 10,
        rerank_top_n: int = 3,
    ) -> list[tuple[str, float]]:
        """Retrieve top chunks with scores.  Reranks if a reranker is available."""
        q_emb = self.embed(query)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        sims = self.embeddings @ q_emb / norms

        k = min(initial_k, len(self.chunks))
        top_idxs = np.argsort(sims)[-k:][::-1]
        candidates = [self.chunks[i] for i in top_idxs]
        candidate_scores = [float(sims[i]) for i in top_idxs]

        if self._reranker is not None and len(candidates) > rerank_top_n:
            return self._reranker.rerank(query, candidates, top_n=rerank_top_n)

        return list(zip(candidates[:rerank_top_n], candidate_scores[:rerank_top_n]))

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Return the best-matching chunks for *query* as a single string."""
        ranked = self.retrieve_chunks(
            query,
            initial_k=max(top_k * 3, 10),
            rerank_top_n=top_k,
        )
        return " ".join(text for text, _score in ranked)

    def retrieve_all_above_threshold(
        self,
        query: str,
        min_similarity: float = 0.3,
        max_chunks: int = 20,
    ) -> list[tuple[str, float]]:
        """Return all chunks with similarity >= *min_similarity*, capped at *max_chunks*."""
        q_emb = self.embed(query)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        sims = self.embeddings @ q_emb / norms

        mask = sims >= min_similarity
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            best = int(np.argmax(sims))
            return [(self.chunks[best], float(sims[best]))]

        ranked_idxs = idxs[np.argsort(sims[idxs])[::-1]][:max_chunks]
        return [(self.chunks[int(i)], float(sims[i])) for i in ranked_idxs]

    @property
    def text_preview(self) -> str:
        """First 300 chars of the corpus (for display)."""
        full = " ".join(self.chunks)
        return full[:300] + ("..." if len(full) > 300 else "")

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def sentence_count(self) -> int:
        """Backward-compat alias."""
        return self.chunk_count


# --- Default store built from the bundled dataset ---
from data import BRIEF, DATA  # noqa: E402

default_store = DocumentStore(raw_text=BRIEF)

# Backward-compat aliases used by retriever_minimal / retriever_numpy
EMBEDDINGS = default_store.embeddings


def embed(text: str) -> np.ndarray:
    return default_store.embed(text)
