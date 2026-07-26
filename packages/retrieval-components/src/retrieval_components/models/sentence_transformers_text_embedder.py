"""Query-aware Sentence Transformers text embedder."""

from __future__ import annotations

from typing import Any

from haystack import component
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder as HaystackSentenceTransformersTextEmbedder,
)

from retrieval_components.dataclasses import Query


@component
class SentenceTransformersTextEmbedder(HaystackSentenceTransformersTextEmbedder):
    """Embed the materialized content of a Query."""

    @component.output_types(embedding=list[float])
    def run(self, query: Query) -> dict[str, Any]:
        if query.content is None:
            raise ValueError(f"Query {query.id!r} has no materialized content.")
        return HaystackSentenceTransformersTextEmbedder.run(self, text=query.content)
