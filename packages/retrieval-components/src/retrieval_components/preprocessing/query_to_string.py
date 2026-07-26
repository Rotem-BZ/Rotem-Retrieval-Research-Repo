"""Convert the shared Query value to a plain string."""

from __future__ import annotations

from haystack import component

from retrieval_components.dataclasses import Query


@component
class QueryToString:
    """Expose materialized query content as a plain text string."""

    @component.output_types(text=str)
    def run(self, query: Query) -> dict[str, str]:
        if query.content is None:
            raise ValueError(f"Query {query.id!r} has no materialized content.")
        return {"text": query.content}
