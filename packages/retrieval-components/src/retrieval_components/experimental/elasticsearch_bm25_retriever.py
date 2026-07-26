"""Elasticsearch BM25 retrieval component."""

from __future__ import annotations

from typing import Any

from haystack import Document, component
from haystack.lazy_imports import LazyImport

from retrieval_components.dataclasses import Query

with LazyImport(
    "Run 'pip install \"retrieval-components[elasticsearch]\"' to use Elasticsearch components"
) as elasticsearch_import:
    from elasticsearch import Elasticsearch


@component
class ElasticsearchBM25Retriever:
    """Retrieve documents from Elasticsearch with a text match query."""

    def __init__(
        self,
        index_name: str,
        hosts: str | list[str] | None = None,
        content_field_name: str = "content",
        meta_field: str = "meta",
        top_k: int = 10,
        client: Any | None = None,
    ) -> None:
        self.index_name = index_name
        self.hosts = hosts
        self.content_field_name = content_field_name
        self.meta_field = meta_field
        self.top_k = top_k
        self._client = client

    def warm_up(self) -> None:
        """Initialize the Elasticsearch client once."""
        if self._client is not None:
            return
        elasticsearch_import.check()
        self._client = Elasticsearch(self.hosts or "http://localhost:9200")

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: Query,
        top_k: int | None = None,
        candidate_document_ids: list[str] | None = None,
    ) -> dict[str, list[Document]]:
        if self._client is None:
            raise RuntimeError("ElasticsearchBM25Retriever must be warmed up before run().")
        if query.content is None:
            raise ValueError("ElasticsearchBM25Retriever requires query content.")

        limit = self.top_k if top_k is None else top_k
        search_query: dict[str, Any] = {"match": {self.content_field_name: query.content}}
        if candidate_document_ids is not None:
            search_query = {
                "bool": {
                    "must": [search_query],
                    "filter": [
                        {
                            "bool": {
                                "should": [
                                    {"ids": {"values": candidate_document_ids}},
                                    {
                                        "terms": {
                                            f"{self.meta_field}.source_document_id": (
                                                candidate_document_ids
                                            )
                                        }
                                    },
                                    {
                                        "terms": {
                                            f"{self.meta_field}.source_document_id.keyword": (
                                                candidate_document_ids
                                            )
                                        }
                                    },
                                ],
                                "minimum_should_match": 1,
                            }
                        }
                    ],
                }
            }

        response = self._client.search(
            index=self.index_name,
            query=search_query,
            size=limit,
        )
        documents = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            documents.append(
                Document(
                    id=hit["_id"],
                    content=source[self.content_field_name],
                    meta=dict(source[self.meta_field]),
                    score=hit["_score"],
                    embedding=source.get("embedding"),
                )
            )
        return {"documents": documents}
