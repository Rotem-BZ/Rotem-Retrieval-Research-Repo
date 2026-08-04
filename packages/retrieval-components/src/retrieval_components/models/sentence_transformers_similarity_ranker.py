"""Query-aware Sentence Transformers similarity ranker."""

from __future__ import annotations

from haystack import Document, component
from haystack.components.rankers import (
    SentenceTransformersSimilarityRanker as HaystackSentenceTransformersSimilarityRanker,
)

from retrieval_components.dataclasses.query import Query


@component
class SentenceTransformersSimilarityRanker(HaystackSentenceTransformersSimilarityRanker):
    """Rank documents against the materialized content of a Query."""

    @component.output_types(documents=list[Document])
    def run(
        self,
        *,
        query: Query,
        documents: list[Document],
        top_k: int | None = None,
        scale_score: bool | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, list[Document]]:
        if query.content is None:
            raise ValueError(f"Query {query.id!r} has no materialized content.")
        return HaystackSentenceTransformersSimilarityRanker.run(
            self,
            query=query.content,
            documents=documents,
            top_k=top_k,
            scale_score=scale_score,
            score_threshold=score_threshold,
        )
