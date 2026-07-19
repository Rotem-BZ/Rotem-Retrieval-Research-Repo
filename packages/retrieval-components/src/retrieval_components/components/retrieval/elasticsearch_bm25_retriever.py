"""Elasticsearch BM25 retrieval component."""

from __future__ import annotations

from typing import Any

from haystack import Document, component

from retrieval_components.utils.elasticsearch import create_client


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
    def run(
        self,
        query: str,
        top_k: int | None = None,
        candidate_document_ids: list[str] | None = None,
    ) -> dict[str, list[Document]]:
        limit = top_k or self.top_k
        search_query: dict[str, Any] = {"match": {self.content_field: query}}
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
        if self._client is None:
            self._client = create_client(self.hosts)
        response = self._client.search(
            index=self.index_name,
            query=search_query,
            size=limit,
        )
        return {
            "documents": [
                Document(
                    id=hit.get("_id") or hit.get("_source", {}).get("id"),
                    content=hit.get("_source", {}).get(self.content_field, ""),
                    meta=dict(hit.get("_source", {}).get(self.meta_field) or {}),
                    score=hit.get("_score"),
                    embedding=hit.get("_source", {}).get("embedding"),
                )
                for hit in response.get("hits", {}).get("hits", [])
            ]
        }
