# AI usage

<!-- Replace every comment below with your own words. Short is fine. Your Track Mentor reads it. -->

## Tools I used, and for what

I used Codex to implement the normaliser, offset-preserving chunker, BM25 index, hybrid retriever,
citation helpers, and tests. I also used it to run deterministic randomized chunking checks and the
12-query retrieval evaluation, then manually reviewed the formulas, offsets, and report values.

## Where AI-written chunking or BM25 code broke one of the spec's rules

The first chunking approach I considered used `end - overlap` directly for every next start. A
counter-example with a sentence cut showed that this can start the next chunk in the middle of a word
even though the preceding cut was structural, violating rule 6. I changed it to select the earliest
word boundary inside the allowed overlap, and a deterministic 1,000-case property check verified the
offset, coverage, overlap, and mid-word invariants.

## My recall@3 table (from EVAL.md) and why hybrid did or did not win

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 11/12 (0.917) | q12 |
| Dense only (HashingEmbedder) | 11/12 (0.917) | q12 |
| Hybrid (RRF) | 11/12 (0.917) | q12 |

Hybrid did not win because both component retrievers were distracted by similar governing-law
boilerplate in other contracts, and the hashing embedder adds spelling overlap rather than semantic
understanding.
