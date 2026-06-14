"""JSONL-backed embedding retrieval component."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haystack import Document, component
import numpy as np


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
        self._index: _EmbeddingIndex | None = None

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        limit = top_k or self.top_k
        index = self._load_index()
        if not index.documents or limit <= 0:
            return {"documents": []}

        scores = _scores(
            query_embedding=query_embedding,
            embeddings=index.embeddings,
            embedding_norms=index.embedding_norms,
            similarity=self.similarity,
        )
        top_indices = _top_indices(scores, limit)
        return {
            "documents": [
                _copy_with_score(index.documents[index_value], float(scores[index_value]))
                for index_value in top_indices
            ]
        }

    def _load_index(self) -> "_EmbeddingIndex":
        if self._index is not None:
            return self._index

        path = Path(self.index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding index not found at {path}. Run the indexing stage first."
            )

        documents: list[Document] = []
        embeddings: list[list[float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    document = _document_from_record(json.loads(line))
                    embedding = getattr(document, "embedding", None)
                    if embedding is not None:
                        documents.append(document)
                        embeddings.append(list(embedding))

        embedding_matrix = _embedding_matrix(embeddings)
        self._index = _EmbeddingIndex(
            documents=documents,
            embeddings=embedding_matrix,
            embedding_norms=np.linalg.norm(embedding_matrix, axis=1),
        )
        return self._index


def _document_from_record(record: dict[str, Any]) -> Document:
    return Document(
        id=record.get("id"),
        content=record.get("content", ""),
        meta=dict(record.get("meta") or {}),
        score=record.get("score"),
        embedding=record.get("embedding"),
    )


@dataclass(frozen=True)
class _EmbeddingIndex:
    documents: list[Document]
    embeddings: np.ndarray
    embedding_norms: np.ndarray


def _embedding_matrix(embeddings: list[list[float]]) -> np.ndarray:
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32)


def _scores(
    *,
    query_embedding: list[float],
    embeddings: np.ndarray,
    embedding_norms: np.ndarray,
    similarity: str,
) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    if embeddings.shape[1] != query.shape[0]:
        raise ValueError(
            "Embedding dimensions differ: "
            f"query has {query.shape[0]} values, index has {embeddings.shape[1]}."
        )

    if similarity == "dot_product":
        return embeddings @ query
    if similarity == "cosine":
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(embeddings.shape[0], dtype=np.float32)
        denominator = embedding_norms * query_norm
        return np.divide(
            embeddings @ query,
            denominator,
            out=np.zeros(embeddings.shape[0], dtype=np.float32),
            where=denominator != 0,
        )

    raise ValueError(f"Unsupported similarity: {similarity}")


def _top_indices(scores: np.ndarray, limit: int) -> list[int]:
    limit = min(limit, scores.shape[0])
    if limit == scores.shape[0]:
        candidate_indices = np.arange(scores.shape[0])
    else:
        candidate_indices = np.argpartition(scores, -limit)[-limit:]
    return sorted(candidate_indices.tolist(), key=lambda index: float(scores[index]), reverse=True)


def _copy_with_score(document: Document, score: float) -> Document:
    return Document(
        id=document.id,
        content=document.content,
        meta=dict(document.meta or {}),
        score=score,
        embedding=getattr(document, "embedding", None),
    )
