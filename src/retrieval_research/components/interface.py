"""Pipeline boundary components used by stage runners."""

from __future__ import annotations

from haystack import Document, component


@component
class InferenceInput:
    """Expose the fixed inference stage inputs as Haystack output sockets."""

    @component.output_types(
        query=str,
        candidate_document_ids=list[str],
        candidate_documents=list[Document],
    )
    def run(
        self,
        query: str,
        candidate_document_ids: list[str] | None = None,
        candidate_documents: list[Document] | None = None,
    ) -> dict[str, str | list[str] | list[Document]]:
        return {
            "query": query,
            "candidate_document_ids": list(candidate_document_ids or []),
            "candidate_documents": list(candidate_documents or []),
        }


@component
class InferenceOutput:
    """Collect the fixed inference stage output."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        return {"documents": documents}


@component
class IndexingOutput:
    """Collect the fixed indexing stage output."""

    @component.output_types(index_path=str, indexed_count=int)
    def run(self, index_path: str, indexed_count: int) -> dict[str, str | int]:
        return {
            "index_path": index_path,
            "indexed_count": indexed_count,
        }
