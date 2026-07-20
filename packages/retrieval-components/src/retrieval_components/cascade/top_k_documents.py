"""Top-k document selector."""

from __future__ import annotations

from haystack import Document, component


def _sort_documents_by_score(documents: list[Document]) -> list[Document]:
    return sorted(
        documents,
        key=lambda document: (float(document.score or 0.0), document.id or ""),
        reverse=True,
    )


@component
class TopKDocuments:
    """Keep the first `top_k` documents after optional score sorting."""

    def __init__(self, top_k: int, sort_by_score: bool = True) -> None:
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        ranked = _sort_documents_by_score(documents) if self.sort_by_score else list(documents)
        return {"documents": ranked[: self.top_k]}
