"""BM25, written by hand. You write this file (no search libraries)."""
from __future__ import annotations

import math
from collections import Counter

from bayyina.types import Chunk
from bayyina.normalize import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        total_length = 0
        for chunk in chunks:
            terms = Counter(tokenize(chunk.text))
            self._chunks[chunk.chunk_id] = terms
            length = sum(terms.values())
            self._lengths[chunk.chunk_id] = length
            total_length += length
            self._document_frequency.update(terms.keys())
        if self._chunks:
            self._average_length = total_length / len(self._chunks)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """[(chunk_id, score), ...] with score > 0, by score descending then chunk ID ascending."""
        query_terms = set(tokenize(query))
        if not query_terms or not self._chunks or top_k <= 0 or self._average_length == 0:
            return []

        scored_chunks: list[tuple[str, float]] = []
        for chunk_id, frequencies in self._chunks.items():
            score = self._score(query_terms, frequencies, self._lengths[chunk_id])
            if score > 0:
                scored_chunks.append((chunk_id, score))

        scored_chunks.sort(key=lambda scored: (-scored[1], scored[0]))
        return scored_chunks[:top_k]

    def _score(
        self,
        query_terms: set[str],
        frequencies: Counter[str],
        document_length: int,
    ) -> float:
        score = 0.0
        n_chunks = len(self._chunks)
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self._document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (n_chunks - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * document_length / self._average_length
            )
            score += inverse_document_frequency * frequency * (self.k1 + 1) / denominator
        return score
