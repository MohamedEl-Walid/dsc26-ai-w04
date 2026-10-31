"""Reciprocal rank fusion and the hybrid retriever. You write this file."""
from __future__ import annotations

import numpy as np

from bayyina.bm25 import BM25Index
from bayyina.types import Chunk, Embedder, Hit, Reranker


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Ranks start at 1; score = sum of 1 / (k + rank); by score descending, then ID ascending."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, *, k1: float = 1.5,
                 b: float = 0.75, reranker: Reranker | None = None):
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._chunk_ids = list(self._chunks)
        self._embedder = embedder
        self._reranker = reranker
        self._bm25 = BM25Index(k1=k1, b=b)
        self._bm25.add(chunks)
        self._embeddings = np.asarray(embedder.embed([chunk.text for chunk in chunks]))

    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
        if not query.strip() or top_k <= 0 or candidates <= 0 or not self._chunks:
            return []

        lexical_ids = [
            chunk_id for chunk_id, _ in self._bm25.search(query, top_k=candidates)
        ]

        dense_ids = self._dense_ranking(query, candidates)

        fused = rrf([lexical_ids, dense_ids])[:candidates]
        hits = [self._make_hit(chunk_id, score) for chunk_id, score in fused]

        if self._reranker is not None:
            hits = self._rerank(query, hits, self._reranker)

        return hits[:top_k]

    def _dense_ranking(self, query: str, candidates: int) -> list[str]:
        query_embedding = np.asarray(self._embedder.embed([query]))[0]
        dense_scores = self._embeddings @ query_embedding
        scored_chunks = sorted(
            zip(self._chunk_ids, dense_scores, strict=True),
            key=lambda scored: (-float(scored[1]), scored[0]),
        )
        return [chunk_id for chunk_id, _ in scored_chunks[:candidates]]

    def _rerank(self, query: str, hits: list[Hit], reranker: Reranker) -> list[Hit]:
        reranker_scores = reranker(query, hits)
        reranked = [
            self._make_hit(hit.chunk_id, float(score))
            for hit, score in zip(hits, reranker_scores, strict=True)
        ]
        reranked.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return reranked

    def _make_hit(self, chunk_id: str, score: float) -> Hit:
        chunk = self._chunks[chunk_id]
        return Hit(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            start=chunk.start,
            end=chunk.end,
            text=chunk.text,
            score=float(score),
        )
