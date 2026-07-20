"""Rank embedded documents by similarity to a query embedding."""

from __future__ import annotations

from haystack import Document, component
import numpy as np


def _copy_document_with_score(document: Document, score: float) -> Document:
    return Document(
        id=document.id,
        content=document.content,
        meta=dict(document.meta or {}),
        score=score,
        embedding=getattr(document, "embedding", None),
    )


def _document_embedding_matrix(documents: list[Document]) -> np.ndarray:
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


def _similarity_scores(
    *,
    query_embedding: list[float],
    embeddings: np.ndarray,
    similarity: str,
) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != query.shape[0]:
        raise ValueError(
            "Embedding dimensions differ: "
            f"query has {query.shape[0]} values, documents have shape {embeddings.shape}."
        )

    if similarity == "dot_product":
        return embeddings @ query
    if similarity == "cosine":
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(embeddings.shape[0], dtype=np.float32)
        denominator = np.linalg.norm(embeddings, axis=1) * query_norm
        return np.divide(
            embeddings @ query,
            denominator,
            out=np.zeros(embeddings.shape[0], dtype=np.float32),
            where=denominator != 0,
        )

    raise ValueError(f"Unsupported similarity: {similarity}")


def _top_score_indices(scores: np.ndarray, limit: int | None) -> list[int]:
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


@component
class EmbeddingSimilarityRanker:
    """Score documents that already carry embeddings against a query embedding."""

    def __init__(
        self,
        top_k: int | None = 10,
        similarity: str = "cosine",
    ) -> None:
        self.top_k = top_k
        self.similarity = similarity

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        documents: list[Document],
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        limit = self.top_k if top_k is None else top_k
        if not documents or limit == 0:
            return {"documents": []}

        embeddings = _document_embedding_matrix(documents)
        scores = _similarity_scores(
            query_embedding=query_embedding,
            embeddings=embeddings,
            similarity=self.similarity,
        )
        ranked_indices = _top_score_indices(scores, limit)
        return {
            "documents": [
                _copy_document_with_score(documents[index], float(scores[index]))
                for index in ranked_indices
            ]
        }
