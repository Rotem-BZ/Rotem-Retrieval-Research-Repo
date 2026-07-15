"""Embedding matrix and similarity helpers."""

from __future__ import annotations

from haystack import Document
import numpy as np


def embedding_matrix(embeddings: list[list[float]]) -> np.ndarray:
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32)


def document_embedding_matrix(documents: list[Document]) -> np.ndarray:
    embeddings: list[list[float]] = []
    missing_ids: list[str | None] = []
    for document in documents:
        embedding = getattr(document, "embedding", None)
        if embedding is None:
            missing_ids.append(document.id)
        else:
            embeddings.append(list(embedding))

    if missing_ids:
        raise ValueError(f"Documents are missing embeddings: {missing_ids}")

    return np.asarray(embeddings, dtype=np.float32)


def similarity_scores(
    *,
    query_embedding: list[float],
    embeddings: np.ndarray,
    similarity: str,
    embedding_norms: np.ndarray | None = None,
    dimension_error_context: str = "embeddings",
) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != query.shape[0]:
        raise ValueError(
            "Embedding dimensions differ: "
            f"query has {query.shape[0]} values, {dimension_error_context} have shape "
            f"{embeddings.shape}."
        )

    if similarity == "dot_product":
        return embeddings @ query
    if similarity == "cosine":
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(embeddings.shape[0], dtype=np.float32)
        norms = embedding_norms
        if norms is None:
            norms = np.linalg.norm(embeddings, axis=1)
        denominator = norms * query_norm
        return np.divide(
            embeddings @ query,
            denominator,
            out=np.zeros(embeddings.shape[0], dtype=np.float32),
            where=denominator != 0,
        )

    raise ValueError(f"Unsupported similarity: {similarity}")


def top_score_indices(scores: np.ndarray, limit: int | None) -> list[int]:
    if limit is None:
        limit = scores.shape[0]
    else:
        limit = min(limit, scores.shape[0])
    if limit <= 0:
        return []
    if limit == scores.shape[0]:
        candidate_indices = np.arange(scores.shape[0])
    else:
        candidate_indices = np.argpartition(scores, -limit)[-limit:]
    return sorted(candidate_indices.tolist(), key=lambda index: float(scores[index]), reverse=True)

