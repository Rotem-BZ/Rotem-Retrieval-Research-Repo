"""Pipeline boundary components for the inference stage."""

from __future__ import annotations

from haystack import Document, component

from retrieval_components.dataclasses import Query


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
