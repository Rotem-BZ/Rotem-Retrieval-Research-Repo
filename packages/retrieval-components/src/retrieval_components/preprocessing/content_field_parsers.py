"""Base components that materialize text from a configured metadata field."""

from __future__ import annotations

from haystack import Document, component

from retrieval_components.dataclasses import Query


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
            meta = dict(document.meta)
            if self.content_field not in meta:
                raise ValueError(
                    f"Document {document.id!r} is missing configured content field "
                    f"{self.content_field!r}."
                )
            parsed_documents.append(
                Document(
                    id=document.id,
                    content=str(meta[self.content_field]),
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

    @component.output_types(query=Query)
    def run(self, query: Query) -> dict[str, Query]:
        if self.content_field not in query.meta:
            raise ValueError(
                f"Query {query.id!r} is missing configured content field {self.content_field!r}."
            )
        return {"query": query.with_content(str(query.meta[self.content_field]))}
