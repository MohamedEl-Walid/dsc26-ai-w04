import math

import numpy as np
import pytest

from bayyina.bm25 import BM25Index
from bayyina.chunking import chunk_text
from bayyina.cite import format_citation, verify_citation
from bayyina.hybrid import HybridRetriever, rrf
from bayyina.normalize import normalize_ar, tokenize
from bayyina.types import Chunk, Hit


def test_normalization_applies_only_the_specified_rules():
    assert normalize_ar("أ إ آ ٱ ى ة ٠١٢ ۰۱۲ مُـدّة") == "ا ا ا ا ي ه 012 012 مده"
    assert normalize_ar("ؤ ئ") == "ؤ ئ"
    assert tokenize("عقد، iPhone sha7en_2 ٣") == ["عقد", "iphone", "sha7en_2", "3"]


@pytest.mark.parametrize(
    ("max_chars", "overlap"),
    [(1, 0), (8, 0), (8, 3), (31, 12)],
)
def test_chunking_invariants(max_chars, overlap):
    text = "  alpha beta.\n\nغرامة التأخير؟\nOneVeryLongWordWithoutBreaks" * 4 + "  "
    chunks = chunk_text("doc", text, max_chars=max_chars, overlap=overlap)

    assert [chunk.chunk_id for chunk in chunks] == [f"doc#{i}" for i in range(len(chunks))]
    assert all(text[c.start:c.end] == c.text for c in chunks)
    assert all(c.text == c.text.strip() and 0 < len(c.text) <= max_chars for c in chunks)
    assert all(a.start < b.start for a, b in zip(chunks, chunks[1:]))
    assert all(max(0, a.end - b.start) <= overlap for a, b in zip(chunks, chunks[1:]))

    for index, character in enumerate(text):
        if not character.isspace():
            assert any(c.start <= index < c.end for c in chunks)

    for previous, current in zip(chunks, chunks[1:]):
        assert not any(not ch.isspace() for ch in text[previous.end:current.start])
        starts_mid_word = current.start > 0 and not text[current.start - 1].isspace()
        previous_cut_mid_word = (
            previous.end < len(text)
            and not text[previous.end - 1].isspace()
            and not text[previous.end].isspace()
        )
        assert not starts_mid_word or previous_cut_mid_word


def test_chunking_cut_priority_and_validation():
    text = "a" * 28 + "\n\n" + "b" * 8 + ". " + "c" * 30
    first = chunk_text("d", text, max_chars=50, overlap=0)[0]
    assert first.text == "a" * 28
    assert chunk_text("d", " \n\t ") == []
    with pytest.raises(ValueError):
        chunk_text("d", "x", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("d", "x", max_chars=5, overlap=5)


def test_bm25_matches_formula_and_ignores_repeated_query_terms():
    chunks = [
        Chunk("d", "d#0", 0, 7, "red red"),
        Chunk("d", "d#1", 8, 16, "red blue"),
    ]
    index = BM25Index(k1=1.5, b=0.75)
    index.add(chunks)
    result = index.search("red red", top_k=2)

    idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    expected = idf * 2 * 2.5 / (2 + 1.5)
    assert result[0] == pytest.approx(("d#0", expected))
    assert result[1][0] == "d#1"
    assert index.search("", 2) == []
    assert index.search("red", 0) == []


def test_rrf_deduplicates_each_ranking():
    assert rrf([["a", "a"], ["b"]], k=10) == [
        ("a", pytest.approx(1 / 11)),
        ("b", pytest.approx(1 / 11)),
    ]


class LookupEmbedder:
    def embed(self, texts):
        vectors = {
            "lexical apple": [1.0, 0.0],
            "dense target": [0.0, 1.0],
            "apple": [1.0, 0.0],
            "needle": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=float)


def test_hybrid_dense_path_offsets_and_reranker():
    chunks = [
        Chunk("d", "d#0", 0, 13, "lexical apple"),
        Chunk("d", "d#1", 14, 26, "dense target"),
    ]
    calls = []

    def reranker(query, hits):
        calls.append((query, hits))
        return [0.1 if hit.chunk_id == "d#0" else 0.9 for hit in hits]

    hits = HybridRetriever(chunks, LookupEmbedder(), reranker=reranker).search(
        "needle", top_k=2, candidates=2
    )
    assert len(calls) == 1
    assert [hit.chunk_id for hit in hits] == ["d#1", "d#0"]
    assert hits[0] == Hit("d#1", "d", 14, 26, "dense target", 0.9)


def test_citation_boundaries_whitespace_and_formatting():
    document = "alpha\n beta gamma"
    assert verify_citation("alpha beta", document, 0, 11)
    assert not verify_citation(" ", document, 0, 11)
    assert not verify_citation("alpha", document, -1, 5)

    hit = Hit("d#0", "d", 0, len(document), document, 1.0)
    assert format_citation(hit, max_quote=11) == '[d:0-17] "alpha beta…"'
