# Bayyina retrieval baseline

## Commands

- Install the locked environment with `uv sync`.
- Run the full verification suite with `uv run pytest`.
- Run no network-dependent checks: the grader runs offline on Python 3.12.

## Project boundaries

- Only the Python standard library, numpy, pandas, and scikit-learn are approved.
- Do not add search libraries such as FAISS or rank_bm25, or change `pyproject.toml` or `uv.lock`.
- Keep `bayyina/types.py` and `bayyina/embedder.py` unchanged: the grader restores them.

## Retrieval and citations

- Preserve original-text offsets: every chunk and hit must satisfy `text[start:end] == text`.
- Normalize Arabic only through `bayyina.normalize.normalize_ar`; do not normalize source text used for citations.
- Use `chunk_text(..., max_chars=500, overlap=50)` for the documented retrieval evaluation.
- Keep BM25 ranking deterministic: score descending, then plain string chunk-ID ascending.
- Treat blank queries and empty indexes as valid cases that return an empty result.
- Cite spans only when `verify_citation` confirms the quote is within the original document range.

## Data, models, and reports

- `data/contracts/` and `data/queries.jsonl` contain synthetic data only; do not add personal or contract data.
- `HashingEmbedder(dim=512, n=3)` is the approved offline baseline; do not claim semantic-model results for it.
- Record benchmark outcomes and misses in `EVAL.md`; keep AI-tool disclosure and validation notes in `AI_USAGE.md`.
- Put any future user-facing retrieval prompts in `prompts/` with a corresponding test before changing them.

## Before completion

- Add a focused test for changed retrieval behavior, including Arabic/mixed-script or offset edge cases where relevant.
- Run `uv run pytest` and report the result rather than claiming success without the command output.
