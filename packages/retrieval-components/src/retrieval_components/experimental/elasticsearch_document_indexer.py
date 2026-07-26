"""Elasticsearch document indexing component."""

from __future__ import annotations

from typing import Any

from haystack import Document, component
from haystack.lazy_imports import LazyImport

with LazyImport(
    "Run 'pip install \"retrieval-components[elasticsearch]\"' to use Elasticsearch components"
) as elasticsearch_import:
    from elasticsearch import Elasticsearch


@component
class ElasticsearchDocumentIndexer:
    """Index Haystack documents into Elasticsearch."""

    def __init__(
        self,
        index_name: str,
        hosts: str | list[str] | None = None,
        content_field_name: str = "content",
        meta_field: str = "meta",
        refresh: bool = False,
        client: Any | None = None,
    ) -> None:
        self.index_name = index_name
        self.hosts = hosts
        self.content_field_name = content_field_name
        self.meta_field = meta_field
        self.refresh = refresh
        self._client = client

    def warm_up(self) -> None:
        """Initialize the Elasticsearch client once."""
        if self._client is not None:
            return
        elasticsearch_import.check()
        self._client = Elasticsearch(self.hosts or "http://localhost:9200")

    @component.output_types(indexed_count=int)
    def run(self, documents: list[Document]) -> dict[str, int]:
        if self._client is None:
            raise RuntimeError("ElasticsearchDocumentIndexer must be warmed up before run().")

        for document in documents:
            source = {
                self.content_field_name: document.content,
                self.meta_field: dict(document.meta),
            }
            if document.embedding is not None:
                source["embedding"] = document.embedding
            self._client.index(
                index=self.index_name,
                id=document.id,
                document=source,
                refresh=self.refresh,
            )

        return {"indexed_count": len(documents)}
