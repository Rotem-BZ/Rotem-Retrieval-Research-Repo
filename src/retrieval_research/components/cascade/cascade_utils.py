"""Shared helpers for cascade selectors."""

from __future__ import annotations

from haystack import Document


def score(document: Document) -> float:
    return float(document.score or 0.0)


def rank(documents: list[Document], sort_by_score: bool) -> list[Document]:
    if not sort_by_score:
        return list(documents)
    return sorted(documents, key=lambda document: (score(document), document.id or ""), reverse=True)
