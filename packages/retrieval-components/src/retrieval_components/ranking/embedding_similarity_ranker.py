"""Rank embedded documents by similarity to a query embedding."""

from __future__ import annotations

import numpy as np
from haystack import Document, component


def _document_embedding_matrix(documents: list[Document]) -> np.ndarray:
    embeddings: list[list[float]] = []
    missing_ids: list[str | None] = []
    for document in documents:
        embedding = document.embedding
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


@component
class EmbeddingSimilarityRanker:
    """Score documents that already carry embeddings against a query embedding."""

    def __init__(
        self,
        similarity: str = "cosine",
    ) -> None:
        self.similarity = similarity

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        documents: list[Document],
    ) -> dict[str, list[Document]]:
        if not documents:
            return {"documents": []}

        embeddings = _document_embedding_matrix(documents)
        scores = _similarity_scores(
            query_embedding=query_embedding,
            embeddings=embeddings,
            similarity=self.similarity,
        )
        ranked_indices = sorted(
            range(scores.shape[0]),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        return {
            "documents": [
                Document(
                    id=documents[index].id,
                    content=documents[index].content,
                    meta=dict(documents[index].meta),
                    score=float(scores[index]),
                    embedding=documents[index].embedding,
                )
                for index in ranked_indices
            ]
        }
