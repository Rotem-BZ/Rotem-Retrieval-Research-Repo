"""Pipeline boundary components used by stage runners."""

from __future__ import annotations

from haystack import Document, component

from retrieval_components.dataclasses import Query


@component
class IndexingInput:
    """Expose the fixed indexing stage input as a Haystack output socket."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        return {"documents": documents}


@component
class InferenceInput:
    """Expose the fixed inference stage inputs as Haystack output sockets."""

    @component.output_types(
        query=Query,
        candidate_document_ids=list[str],
        candidate_documents=list[Document],
    )
    def run(
        self,
        query: Query,
        candidate_document_ids: list[str] | None = None,
        candidate_documents: list[Document] | None = None,
    ) -> dict[str, Query | list[str] | list[Document]]:
        return {
            "query": query,
            "candidate_document_ids": list(candidate_document_ids or []),
            "candidate_documents": list(candidate_documents or []),
        }


@component
class InferenceOutput:
    """Collect the fixed inference stage output."""

    @component.output_types(documents=list[Document], query=Query)
    def run(
        self,
        documents: list[Document],
        query: Query,
    ) -> dict[str, list[Document] | Query]:
        return {"documents": documents, "query": query}


@component
class IndexingOutput:
    """Collect the fixed indexing stage output."""

    @component.output_types(index_path=str, indexed_count=int)
    def run(self, index_path: str, indexed_count: int) -> dict[str, str | int]:
        return {
            "index_path": index_path,
            "indexed_count": indexed_count,
        }
