"""Canonical typed records for retrieval datasets and prediction artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from haystack import Document
from retrieval_components.dataclasses.query import Query


@dataclass(frozen=True)
class Qrel:
    """A relevance judgment shared by every query representing one information need."""

    IN: str
    document_id: str
    label: int

    def __post_init__(self) -> None:
        if not isinstance(self.IN, str) or not self.IN:
            raise ValueError("Qrel IN must be a non-empty string.")
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("Qrel document_id must be a non-empty string.")
        if isinstance(self.label, bool) or not isinstance(self.label, int):
            raise TypeError("Qrel label must be an integer.")

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Qrel:
        if not isinstance(data, dict):
            raise TypeError("Qrel record must be an object.")
        expected = {field.name for field in fields(cls)}
        missing = expected - data.keys()
        unexpected = data.keys() - expected
        if missing:
            raise ValueError(f"Qrel record is missing required fields: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"Qrel record has unexpected fields: {sorted(unexpected)}")
        return cls(**data)


def query_from_dict(data: dict[str, Any]) -> Query:
    """Validate and construct one canonical dataset query."""

    if not isinstance(data, dict):
        raise TypeError("Query record must be an object.")
    expected = {field.name for field in fields(Query)}
    missing = expected - data.keys()
    unexpected = data.keys() - expected
    if missing:
        raise ValueError(f"Query record is missing required fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Query record has unexpected fields: {sorted(unexpected)}")
    query = Query.from_dict(data)
    if query.content is None:
        raise ValueError("Dataset Query content must be a string.")
    if query.IN is None:
        raise ValueError("Dataset Query IN must be a non-empty string.")
    return query


def document_from_dict(data: dict[str, Any]) -> Document:
    """Validate and construct one canonical dataset document."""

    if not isinstance(data, dict):
        raise TypeError("Document record must be an object.")
    document = Document.from_dict(data)
    if not isinstance(document.id, str) or not document.id:
        raise ValueError("Document id must be a non-empty string.")
    if document.content is None:
        raise ValueError("Dataset Document content must be a string.")
    if not isinstance(document.content, str):
        raise TypeError("Document content must be a string.")
    if not isinstance(document.meta, dict):  # pragma: no cover - enforced by Haystack
        raise TypeError("Document meta must be a dictionary.")
    return document
