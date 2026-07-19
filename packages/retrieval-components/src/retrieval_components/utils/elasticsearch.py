"""Elasticsearch helpers shared by indexing and retrieval components."""

from __future__ import annotations

from typing import Any

def create_client(hosts: str | list[str] | None) -> Any:
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise ImportError(
            "Elasticsearch components require the optional `elasticsearch` package "
            "or an injected client."
        ) from exc

    return Elasticsearch(hosts or "http://localhost:9200")
