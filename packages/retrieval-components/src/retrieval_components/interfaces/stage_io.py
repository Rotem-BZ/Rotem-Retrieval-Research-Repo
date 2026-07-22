"""Pipeline boundary components used by stage runners."""

from __future__ import annotations

from typing import Any

from haystack import Document, component


@component
class InferenceInput:
    """Expose the fixed inference stage inputs as Haystack output sockets."""

    @component.output_types(
        query=str,
        query_meta=dict[str, Any],
        candidate_document_ids=list[str],
        candidate_documents=list[Document],
    )
    def run(
        self,
        query_meta: dict[str, Any] | None = None,
        query: str = "",
        candidate_document_ids: list[str] | None = None,
        candidate_documents: list[Document] | None = None,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "query_meta": dict(query_meta or {}),
            "candidate_document_ids": list(candidate_document_ids or []),
            "candidate_documents": list(candidate_documents or []),
        }


@component
class InferenceOutput:
    """Collect the fixed inference stage output."""

    @component.output_types(documents=list[Document], query_content=str | None)
    def run(
        self,
        documents: list[Document],
        query_content: str | None = None,
    ) -> dict[str, list[Document] | str | None]:
        return {"documents": documents, "query_content": query_content}


@component
class IndexingOutput:
    """Collect the fixed indexing stage output."""

    @component.output_types(index_path=str, indexed_count=int)
    def run(self, index_path: str, indexed_count: int) -> dict[str, str | int]:
        return {
            "index_path": index_path,
            "indexed_count": indexed_count,
        }
