"""Per-source-document chunk selector."""

from __future__ import annotations

from haystack import Document, component


def _source_document_id(document: Document) -> str:
    source_id = (document.meta or {}).get("source_document_id") or document.id
    if source_id is None:
        raise ValueError(
            "ChunkCascade requires every document to define `meta.source_document_id` or `id`."
        )
    return str(source_id)


def _sort_documents_by_score(documents: list[Document]) -> list[Document]:
    return sorted(
        documents,
        key=lambda document: (float(document.score or 0.0), document.id or ""),
        reverse=True,
    )


@component
class ChunkCascade:
    """Keep at most `top_k` chunks for each source document."""

    def __init__(self, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        ranked = _sort_documents_by_score(documents)
        selected: list[Document] = []
        selected_counts: dict[str, int] = {}

        for document in ranked:
            source_id = _source_document_id(document)
            selected_count = selected_counts.get(source_id, 0)
            if selected_count >= self.top_k:
                continue
            selected.append(document)
            selected_counts[source_id] = selected_count + 1

        return {"documents": selected}
