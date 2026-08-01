"""Document parser that validates already-materialized content."""

from __future__ import annotations

from haystack import Document, component


@component
class IdentityParser:
    """Return documents unchanged after verifying that content is present."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        for document in documents:
            if document.content is None:
                raise ValueError(f"Document {document.id!r} is missing content.")
        return {"documents": documents}
