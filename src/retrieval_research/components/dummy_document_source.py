"""Dummy in-memory document source."""

from __future__ import annotations

from typing import Any

from haystack import Document, component

from retrieval_research.components.dummy_utils import document_from_record


@component
class DummyDocumentSource:
    """Emit configured documents into a Haystack pipeline."""

    def __init__(self, documents: list[dict[str, Any]] | None = None) -> None:
        self.documents = documents or []

    @component.output_types(documents=list[Document])
    def run(self) -> dict[str, list[Document]]:
        return {"documents": [document_from_record(record) for record in self.documents]}
