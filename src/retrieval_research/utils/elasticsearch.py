"""Elasticsearch helpers shared by indexing and retrieval components."""

from __future__ import annotations

from typing import Any

from haystack import Document


def create_client(hosts: str | list[str] | None) -> Any:
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise ImportError(
            "Elasticsearch components require the optional `elasticsearch` package "
            "or an injected client."
        ) from exc

    return Elasticsearch(hosts or "http://localhost:9200")


def document_to_source(document: Document, content_field: str, meta_field: str) -> dict[str, Any]:
    source = {
        content_field: document.content,
        meta_field: dict(document.meta or {}),
    }
    embedding = getattr(document, "embedding", None)
    if embedding is not None:
        source["embedding"] = embedding
    return source


def hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    return list(response.get("hits", {}).get("hits", []))


def hit_to_document(hit: dict[str, Any], content_field: str, meta_field: str) -> Document:
    source = hit.get("_source", {})
    return Document(
        id=hit.get("_id") or source.get("id"),
        content=source.get(content_field, ""),
        meta=dict(source.get(meta_field) or {}),
        score=hit.get("_score"),
        embedding=source.get("embedding"),
    )

