"""JSONL-backed embedding retrieval component."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haystack import Document, component
import numpy as np


def _document_from_record(record: dict[str, Any]) -> Document:
    return Document(
        id=record.get("id"),
        content=record.get("content", ""),
        meta=dict(record.get("meta") or {}),
        score=record.get("score"),
        embedding=record.get("embedding"),
    )


def _candidate_document_id(document: Document) -> str | None:
    meta = document.meta or {}
    return meta.get("source_document_id") or document.id


def _copy_document_with_score(document: Document, score: float) -> Document:
    return Document(
        id=document.id,
        content=document.content,
        meta=dict(document.meta or {}),
        score=score,
        embedding=getattr(document, "embedding", None),
    )


def _similarity_scores(
    *,
    query_embedding: list[float],
    embeddings: np.ndarray,
    similarity: str,
    embedding_norms: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != query.shape[0]:
        raise ValueError(
            "Embedding dimensions differ: "
            f"query has {query.shape[0]} values, index embeddings have shape "
            f"{embeddings.shape}."
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


def _top_score_indices(scores: np.ndarray, limit: int) -> list[int]:
    limit = min(limit, scores.shape[0])
    if limit <= 0:
        return []
    if limit == scores.shape[0]:
        candidate_indices = np.arange(scores.shape[0])
    else:
        candidate_indices = np.argpartition(scores, -limit)[-limit:]
    return sorted(candidate_indices.tolist(), key=lambda index: float(scores[index]), reverse=True)


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
        candidate_document_ids: list[str] | None = None,
    ) -> dict[str, list[Document]]:
        limit = top_k or self.top_k
        index = self._load_index()
        if not index.documents or limit <= 0:
            return {"documents": []}

        if candidate_document_ids is None:
            candidate_indices = list(range(len(index.documents)))
        else:
            allowed_ids = set(candidate_document_ids)
            candidate_indices = [
                index
                for index, document in enumerate(index.documents)
                if _candidate_document_id(document) in allowed_ids
            ]
        if not candidate_indices:
            return {"documents": []}

        scores = _similarity_scores(
            query_embedding=query_embedding,
            embeddings=index.embeddings[candidate_indices],
            embedding_norms=index.embedding_norms[candidate_indices],
            similarity=self.similarity,
        )
        top_indices = _top_score_indices(scores, limit)
        return {
            "documents": [
                _copy_document_with_score(
                    index.documents[candidate_indices[index_value]],
                    float(scores[index_value]),
                )
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

        matrix = (
            np.asarray(embeddings, dtype=np.float32)
            if embeddings
            else np.empty((0, 0), dtype=np.float32)
        )
        self._index = _EmbeddingIndex(
            documents=documents,
            embeddings=matrix,
            embedding_norms=np.linalg.norm(matrix, axis=1),
        )
        return self._index


@dataclass(frozen=True)
class _EmbeddingIndex:
    documents: list[Document]
    embeddings: np.ndarray
    embedding_norms: np.ndarray
