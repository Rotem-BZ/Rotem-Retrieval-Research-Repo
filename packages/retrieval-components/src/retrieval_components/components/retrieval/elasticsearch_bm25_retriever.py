"""Elasticsearch BM25 retrieval component."""

from __future__ import annotations

from typing import Any

from haystack import Document, component

from retrieval_components.utils.elasticsearch import (
    create_client,
    hit_to_document,
    hits,
)


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
        response = self._get_client().search(
            index=self.index_name,
            query=search_query,
            size=limit,
        )
        return {
            "documents": [
                hit_to_document(hit, self.content_field, self.meta_field) for hit in hits(response)
            ]
        }

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = create_client(self.hosts)
        return self._client
