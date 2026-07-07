"""Rank embedded documents by similarity to a query embedding."""

from __future__ import annotations

from haystack import Document, component

from retrieval_research.utils.documents import copy_document_with_score
from retrieval_research.utils.embeddings import (
    document_embedding_matrix,
    similarity_scores,
    top_score_indices,
)


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

        embeddings = document_embedding_matrix(documents)
        scores = similarity_scores(
            query_embedding=query_embedding,
            embeddings=embeddings,
            similarity=self.similarity,
            dimension_error_context="documents",
        )
        ranked_indices = top_score_indices(scores, limit)
        return {
            "documents": [
                copy_document_with_score(documents[index], float(scores[index]))
                for index in ranked_indices
            ]
        }
