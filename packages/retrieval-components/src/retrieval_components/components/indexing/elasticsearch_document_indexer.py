"""Elasticsearch document indexing component."""

from __future__ import annotations

from typing import Any

from haystack import Document, component

from retrieval_components.utils.elasticsearch import create_client


@component
class ElasticsearchDocumentIndexer:
    """Index Haystack documents into Elasticsearch."""

    def __init__(
        self,
        index_name: str,
        hosts: str | list[str] | None = None,
        content_field: str = "content",
        meta_field: str = "meta",
        refresh: bool = False,
        client: Any | None = None,
    ) -> None:
        self.index_name = index_name
        self.hosts = hosts
        self.content_field = content_field
        self.meta_field = meta_field
        self.refresh = refresh
        self._client = client

    @component.output_types(indexed_count=int)
    def run(self, documents: list[Document]) -> dict[str, int]:
        if self._client is None:
            self._client = create_client(self.hosts)

        for document in documents:
            source = {
                self.content_field: document.content,
                self.meta_field: dict(document.meta or {}),
            }
            embedding = getattr(document, "embedding", None)
            if embedding is not None:
                source["embedding"] = embedding
            self._client.index(
                index=self.index_name,
                id=document.id,
                document=source,
                refresh=self.refresh,
            )

        return {"indexed_count": len(documents)}
