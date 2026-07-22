"""Base components that materialize text from a configured metadata field."""

from __future__ import annotations

from typing import Any

from haystack import Document, component


def _content_from_meta(meta: dict[str, Any], *, content_field: str, record_label: str) -> str:
    if content_field not in meta:
        raise ValueError(f"{record_label} is missing configured content field {content_field!r}.")
    return str(meta[content_field])


@component
class DocumentContentFieldParser:
    """Set each document's content from one metadata field."""

    def __init__(self, content_field: str) -> None:
        if not content_field.strip():
            raise ValueError("content_field must be a non-empty field name.")
        self.content_field = content_field

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        parsed_documents: list[Document] = []
        for document in documents:
            meta = dict(document.meta or {})
            parsed_documents.append(
                Document(
                    id=document.id,
                    content=_content_from_meta(
                        meta,
                        content_field=self.content_field,
                        record_label=f"Document {document.id!r}",
                    ),
                    blob=document.blob,
                    meta=meta,
                    score=document.score,
                    embedding=document.embedding,
                    sparse_embedding=document.sparse_embedding,
                )
            )
        return {"documents": parsed_documents}


@component
class QueryContentFieldParser:
    """Render query text from one query metadata field."""

    def __init__(self, content_field: str) -> None:
        if not content_field.strip():
            raise ValueError("content_field must be a non-empty field name.")
        self.content_field = content_field

    @component.output_types(text=str)
    def run(self, meta: dict[str, Any]) -> dict[str, str]:
        return {
            "text": _content_from_meta(
                meta,
                content_field=self.content_field,
                record_label="Query",
            )
        }
