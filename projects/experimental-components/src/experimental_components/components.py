"""Small adapters needed to connect third-party components to repository contracts."""

from __future__ import annotations

from dataclasses import replace

from haystack import Document, component


@component
class SourceDocumentIdAdapter:
    """Alias a third-party source-id metadata field to the repository field name."""

    def __init__(
        self,
        source_field: str = "source_id",
        target_field: str = "source_document_id",
    ) -> None:
        if not source_field.strip():
            raise ValueError("source_field must not be empty.")
        if not target_field.strip():
            raise ValueError("target_field must not be empty.")
        if source_field == target_field:
            raise ValueError("source_field and target_field must be different.")
        self.source_field = source_field
        self.target_field = target_field

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        adapted: list[Document] = []
        for document in documents:
            meta = dict(document.meta or {})
            if self.source_field not in meta:
                raise ValueError(
                    f"Document {document.id!r} is missing meta.{self.source_field}; "
                    "the Chonkie output contract may have changed."
                )
            source_id = meta[self.source_field]
            if self.target_field in meta and meta[self.target_field] != source_id:
                raise ValueError(
                    f"Document {document.id!r} has conflicting meta.{self.source_field} "
                    f"and meta.{self.target_field} values."
                )
            meta[self.target_field] = source_id
            adapted.append(replace(document, meta=meta))
        return {"documents": adapted}
