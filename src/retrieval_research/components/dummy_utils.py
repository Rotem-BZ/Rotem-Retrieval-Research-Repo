"""Shared helpers for small JSONL-backed toy components."""

from __future__ import annotations

import json
import re
from collections import Counter
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


def tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))
