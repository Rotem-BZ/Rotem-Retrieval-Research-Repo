"""Document conversion, scoring, and selection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haystack import Document


def document_from_record(record: dict[str, Any] | Document) -> Document:
    if isinstance(record, Document):
        return record

    return Document(
        id=record.get("id"),
        content=record.get("content", ""),
        meta=dict(record.get("meta") or {}),
        score=record.get("score"),
        embedding=record.get("embedding"),
    )


def document_to_record(document: Document) -> dict[str, Any]:
    record = {
        "id": document.id,
        "content": document.content,
        "meta": dict(document.meta or {}),
        "score": getattr(document, "score", None),
    }
    embedding = getattr(document, "embedding", None)
    if embedding is not None:
        if hasattr(embedding, "tolist") and callable(embedding.tolist):
            embedding = embedding.tolist()
        record["embedding"] = embedding
    return record


def read_jsonl_documents(path: str | Path) -> list[Document]:
    resolved = Path(path)
    documents: list[Document] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(document_from_record(json.loads(line)))
    return documents


def copy_document_with_score(document: Document, score: float) -> Document:
    return Document(
        id=document.id,
        content=document.content,
        meta=dict(document.meta or {}),
        score=score,
        embedding=getattr(document, "embedding", None),
    )


def document_score(document: Document) -> float:
    return float(document.score or 0.0)


def sort_documents_by_score(documents: list[Document]) -> list[Document]:
    return sorted(
        documents,
        key=lambda document: (document_score(document), document.id or ""),
        reverse=True,
    )


def candidate_document_id(document: Document) -> str | None:
    meta = document.meta or {}
    return meta.get("source_document_id") or document.id


def filter_documents_by_candidate_ids(
    documents: list[Document],
    candidate_document_ids: list[str] | None,
) -> list[Document]:
    if candidate_document_ids is None:
        return documents

    allowed_ids = set(candidate_document_ids)
    return [document for document in documents if candidate_document_id(document) in allowed_ids]

