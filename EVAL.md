# Retrieval evaluation

<!-- Read by your Track Mentor, not graded by tests. Replace the comments. -->

Setup: all 8 contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`,
`HashingEmbedder()`, the 12 queries in `data/queries.jsonl`. A hit is relevant when its span
overlaps one of the query's `relevant` ranges in the same document.

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 11/12 (0.917) | q12 |
| Dense only (HashingEmbedder) | 11/12 (0.917) | q12 |
| Hybrid (RRF) | 11/12 (0.917) | q12 |

The only miss was q12, whose generic English wording about governing law and a competent court
matched similar boilerplate in several other contracts; BM25 ranked the relevant agency chunk fourth,
while the character-ngram dense retriever did not place it in the top five. Hybrid therefore tied the
single-path recall rather than improving it. I would next try a multilingual semantic embedding model
and contextual chunk headers containing the contract type, then add a small cross-encoder re-ranker.
