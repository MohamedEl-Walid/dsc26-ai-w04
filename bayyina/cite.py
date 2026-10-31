"""Citation checks. You write this file."""
from __future__ import annotations

from bayyina.types import Hit


def verify_citation(quote: str, document: str, start: int, end: int) -> bool:
    if not (0 <= start < end <= len(document)) or not quote.strip():
        return False
    collapsed_quote = " ".join(quote.split())
    collapsed_span = " ".join(document[start:end].split())
    return collapsed_quote in collapsed_span


def format_citation(hit: Hit, max_quote: int = 160) -> str:
    """[doc_id:start-end] "quote" """
    quote = " ".join(hit.text.split())
    if len(quote) > max_quote:
        prefix = quote[:max_quote]
        last_space = prefix.rfind(" ")
        if last_space >= 0:
            prefix = prefix[:last_space]
        quote = f"{prefix}…"
    return f'[{hit.doc_id}:{hit.start}-{hit.end}] "{quote}"'
