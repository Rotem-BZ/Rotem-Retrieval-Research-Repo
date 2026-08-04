"""Query-aware Transformers similarity ranker."""

from __future__ import annotations

from typing import Any

from haystack import Document, component
from haystack.components.rankers import (
    TransformersSimilarityRanker as HaystackTransformersSimilarityRanker,
)

from retrieval_components.dataclasses.query import Query


@component
class TransformersSimilarityRanker(HaystackTransformersSimilarityRanker):
    """Rank documents against the materialized content of a Query."""

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: Query,
        documents: list[Document],
        top_k: int | None = None,
        scale_score: bool | None = None,
        calibration_factor: float | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        if query.content is None:
            raise ValueError(f"Query {query.id!r} has no materialized content.")
        return HaystackTransformersSimilarityRanker.run(
            self,
            query=query.content,
            documents=documents,
            top_k=top_k,
            scale_score=scale_score,
            calibration_factor=calibration_factor,
            score_threshold=score_threshold,
        )
