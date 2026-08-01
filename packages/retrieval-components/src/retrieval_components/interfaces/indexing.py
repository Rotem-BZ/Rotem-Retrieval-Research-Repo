"""Pipeline boundary components for the indexing stage."""

from __future__ import annotations

from haystack import Document, component


@component
class IndexingInput:
    """Expose the fixed indexing stage input as a Haystack output socket."""

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
