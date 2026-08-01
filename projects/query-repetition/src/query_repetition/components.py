"""Project-specific Haystack components."""

from haystack import component
from retrieval_components.dataclasses import Query


@component
class QueryRepeater:
    """Repeat the original query before it is passed to the E5 preprocessor."""

    def __init__(self, separator: str = " ") -> None:
        self.separator = separator

    @component.output_types(query=Query)
    def run(self, query: Query) -> dict[str, Query]:
        if query.content is None:
            raise ValueError(f"Query {query.id!r} has no materialized content.")
        return {"query": query.with_content(self.separator.join((query.content, query.content)))}
