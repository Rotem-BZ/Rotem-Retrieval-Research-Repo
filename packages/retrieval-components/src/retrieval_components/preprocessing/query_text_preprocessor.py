"""Query-aware text preprocessing component."""

from __future__ import annotations

from haystack import component

from retrieval_components.dataclasses import Query
from retrieval_components.preprocessing.text_preprocessor import _apply_text_transforms


@component
class QueryTextPreprocessor:
    """Clean query content while preserving its identity and metadata."""

    def __init__(
        self,
        prefix: str = "",
        suffix: str = "",
        strip: bool = True,
        lowercase: bool = False,
        replace_regexes: dict[str, str] | None = None,
    ) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.strip = strip
        self.lowercase = lowercase
        self.replace_regexes = replace_regexes or {}

    @component.output_types(query=Query)
    def run(self, query: Query) -> dict[str, Query]:
        if query.content is None:
            raise ValueError(f"Query {query.id!r} has no materialized content.")
        return {
            "query": query.with_content(
                _apply_text_transforms(
                    query.content,
                    prefix=self.prefix,
                    suffix=self.suffix,
                    strip=self.strip,
                    lowercase=self.lowercase,
                    replace_regexes=self.replace_regexes,
                )
            )
        }
