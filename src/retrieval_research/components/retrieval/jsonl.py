"""JSONL-backed embedding retrieval."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from haystack import Document, component


@component
class JsonlEmbeddingRetriever:
    """Retrieve documents by comparing a query embedding to JSONL document embeddings."""

    def __init__(
        self,
        index_path: str,
        top_k: int = 10,
        similarity: str = "cosine",
    ) -> None:
        self.index_path = index_path
        self.top_k = top_k
        self.similarity = similarity

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        limit = top_k or self.top_k
        documents = self._load_documents()

        scored: list[Document] = []
        for document in documents:
            embedding = getattr(document, "embedding", None)
            if embedding is None:
                continue
            score = _similarity(query_embedding, list(embedding), self.similarity)
            scored.append(
                Document(
                    id=document.id,
                    content=document.content,
                    meta=dict(document.meta or {}),
                    score=score,
                    embedding=embedding,
                )
            )

        scored.sort(key=lambda document: (document.score or 0.0, document.id or ""), reverse=True)
        return {"documents": scored[:limit]}

    def _load_documents(self) -> list[Document]:
        path = Path(self.index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding index not found at {path}. Run the indexing stage first."
            )

        documents: list[Document] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    documents.append(_document_from_record(json.loads(line)))
        return documents


def _document_from_record(record: dict[str, Any]) -> Document:
    return Document(
        id=record.get("id"),
        content=record.get("content", ""),
        meta=dict(record.get("meta") or {}),
        score=record.get("score"),
        embedding=record.get("embedding"),
    )


def _similarity(left: list[float], right: list[float], similarity: str) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimensions differ: query has {len(left)} values, document has {len(right)}."
        )

    if similarity == "dot_product":
        return sum(a * b for a, b in zip(left, right, strict=True))
    if similarity == "cosine":
        return _cosine(left, right)

    raise ValueError(f"Unsupported similarity: {similarity}")


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
