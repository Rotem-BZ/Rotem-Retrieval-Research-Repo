"""Shared helpers for document fusion components."""

from __future__ import annotations

from haystack import Document


def copy_with_score(document: Document, score: float) -> Document:
    return Document(
        id=document.id,
        content=document.content,
        meta=dict(document.meta or {}),
        score=score,
        embedding=getattr(document, "embedding", None),
    )


def sort_and_limit(documents: list[Document], top_k: int | None) -> list[Document]:
    ranked = sorted(
        documents,
        key=lambda document: (document.score or 0.0, document.id or ""),
        reverse=True,
    )
    if top_k is not None:
        return ranked[:top_k]
    return ranked
