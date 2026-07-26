"""HTTP-backed query reformulation component."""

from __future__ import annotations

from typing import Any

import requests
from haystack import component

from retrieval_components.dataclasses import Query


@component
class HttpQueryReformulator:
    """Call an HTTP service that returns one or more reformulated queries."""

    def __init__(
        self,
        url: str,
        request_field: str = "query",
        response_path: str = "query",
        headers: dict[str, str] | None = None,
        extra_payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.request_field = request_field
        self.response_path = response_path
        self.headers = headers or {}
        self.extra_payload = extra_payload or {}
        self.timeout = timeout

    @component.output_types(query=Query, queries=list[Query])
    def run(self, query: Query) -> dict[str, Query | list[Query]]:
        if query.content is None:
            raise ValueError(f"Query {query.id!r} has no materialized content.")
        payload = dict(self.extra_payload)
        payload[self.request_field] = query.content

        response = requests.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        extracted = response.json()
        for part in self.response_path.split("."):
            if not part:
                continue
            if isinstance(extracted, dict):
                extracted = extracted[part]
            elif isinstance(extracted, list):
                extracted = extracted[int(part)]
            else:
                raise TypeError(
                    f"Cannot extract '{self.response_path}' from non-container response."
                )

        if isinstance(extracted, list):
            queries = [query.with_content(str(item)) for item in extracted]
        else:
            queries = [query.with_content(str(extracted))]

        return {"query": queries[0] if queries else query, "queries": queries}
