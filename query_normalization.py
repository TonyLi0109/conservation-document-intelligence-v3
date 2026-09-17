"""Minimal normalization of user query wrappers before routing and retrieval."""
from __future__ import annotations


def strip_wrapping_query_quotes(query: str) -> str:
    """Remove one balanced pair of quotes only when it encloses the whole query.

    Quotes inside a question remain intact and continue to request literal phrase
    matching. The raw user input remains available in PipelineTracer's
    ``original_query`` event; this function normalizes only downstream behavior.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    text = query.strip()
    pairs = {'"': '"', "\u201c": "\u201d", "\uff02": "\uff02"}
    closing = pairs.get(text[:1])
    if closing and len(text) >= 2 and text.endswith(closing):
        return text[1:-1].strip()
    return text
