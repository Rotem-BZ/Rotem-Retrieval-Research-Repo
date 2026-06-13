"""Elasticsearch indexing and lexical retrieval components."""

from __future__ import annotations

from typing import Any

from haystack import Document, component


def _create_client(hosts: str | list[str] | None) -> Any:
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise ImportError(
            "Elasticsearch components require the optional `elasticsearch` package "
            "or an injected client."
        ) from exc

    return Elasticsearch(hosts or "http://localhost:9200")


def _document_to_source(document: Document, content_field: str, meta_field: str) -> dict[str, Any]:
    source = {
        content_field: document.content,
        meta_field: dict(document.meta or {}),
    }
    embedding = getattr(document, "embedding", None)
    if embedding is not None:
        source["embedding"] = embedding
    return source


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
        client = self._get_client()

        for document in documents:
            client.index(
                index=self.index_name,
                id=document.id,
                document=_document_to_source(document, self.content_field, self.meta_field),
                refresh=self.refresh,
            )

        return {"indexed_count": len(documents)}

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _create_client(self.hosts)
        return self._client


@component
class ElasticsearchBM25Retriever:
    """Retrieve documents from Elasticsearch with a text match query."""

    def __init__(
        self,
        index_name: str,
        hosts: str | list[str] | None = None,
        content_field: str = "content",
        meta_field: str = "meta",
        top_k: int = 10,
        client: Any | None = None,
    ) -> None:
        self.index_name = index_name
        self.hosts = hosts
        self.content_field = content_field
        self.meta_field = meta_field
        self.top_k = top_k
        self._client = client

    @component.output_types(documents=list[Document])
    def run(self, query: str, top_k: int | None = None) -> dict[str, list[Document]]:
        limit = top_k or self.top_k
        response = self._get_client().search(
            index=self.index_name,
            query={"match": {self.content_field: query}},
            size=limit,
        )
        return {"documents": [_hit_to_document(hit, self.content_field, self.meta_field) for hit in _hits(response)]}

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _create_client(self.hosts)
        return self._client


def _hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    return list(response.get("hits", {}).get("hits", []))


def _hit_to_document(hit: dict[str, Any], content_field: str, meta_field: str) -> Document:
    source = hit.get("_source", {})
    return Document(
        id=hit.get("_id") or source.get("id"),
        content=source.get(content_field, ""),
        meta=dict(source.get(meta_field) or {}),
        score=hit.get("_score"),
        embedding=source.get("embedding"),
    )
