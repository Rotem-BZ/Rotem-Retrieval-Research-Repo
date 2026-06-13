"""HTTP-backed query reformulation."""

from __future__ import annotations

from typing import Any

import requests
from haystack import component


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

    @component.output_types(query=str, queries=list[str])
    def run(self, query: str) -> dict[str, str | list[str]]:
        payload = dict(self.extra_payload)
        payload[self.request_field] = query

        response = requests.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        extracted = _extract_path(response.json(), self.response_path)

        if isinstance(extracted, list):
            queries = [str(item) for item in extracted]
        else:
            queries = [str(extracted)]

        return {"query": queries[0] if queries else query, "queries": queries}


def _extract_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise TypeError(f"Cannot extract '{path}' from non-container response.")
    return current
