"""Offset-preserving chunker. You write this file. See README.md for the seven rules."""
from __future__ import annotations

from bayyina.types import Chunk


_SENTENCE_ENDS = frozenset(".!?؟۔\n")


def _last_cut(window: str, minimum: int) -> int:
    """Return the preferred cut offset within a full-size window."""
    paragraph = window.rfind("\n\n", max(0, minimum - 2))
    if paragraph >= 0 and paragraph + 2 >= minimum:
        return paragraph + 2

    for index in range(len(window) - 1, minimum - 2, -1):
        if window[index] in _SENTENCE_ENDS:
            return index + 1

    for index in range(len(window) - 1, minimum - 2, -1):
        if window[index].isspace():
            return index + 1

    return len(window)


def _next_start(text: str, start: int, end: int, overlap: int) -> int:
    """Choose a non-whitespace next start while respecting overlap and words."""
    ended_mid_word = (
        end < len(text)
        and not text[end - 1].isspace()
        and not text[end].isspace()
    )
    lower = max(start + 1, end - overlap)

    if ended_mid_word:
        candidate = lower
        while candidate < len(text) and text[candidate].isspace():
            candidate += 1
        return candidate

    # Prefer the earliest word boundary inside the permitted overlap.
    for candidate in range(lower, min(end, len(text) - 1) + 1):
        if (
            not text[candidate].isspace()
            and (candidate == 0 or text[candidate - 1].isspace())
        ):
            return candidate

    # Zero overlap is always valid; skipping only whitespace preserves coverage.
    candidate = end
    while candidate < len(text) and text[candidate].isspace():
        candidate += 1
    return candidate


def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split `text` into chunks whose offsets point into the ORIGINAL text.

    ValueError unless max_chars > 0 and 0 <= overlap < max_chars.
    """
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("max_chars must be positive and overlap must satisfy 0 <= overlap < max_chars")

    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    if start == len(text):
        return []

    chunks: list[Chunk] = []
    while start < len(text):
        if len(text) - start <= max_chars:
            raw_end = len(text)
        else:
            window = text[start:start + max_chars]
            raw_end = start + _last_cut(window, max_chars // 2)

        end = raw_end
        while end > start and text[end - 1].isspace():
            end -= 1

        # start always points at non-whitespace and a hard cut always advances.
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}#{len(chunks)}",
            start=start,
            end=end,
            text=text[start:end],
        ))

        if raw_end == len(text):
            break
        start = _next_start(text, start, end, overlap)

    return chunks
