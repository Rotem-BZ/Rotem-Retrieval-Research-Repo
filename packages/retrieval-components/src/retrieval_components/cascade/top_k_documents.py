"""Top-k document selector."""

from __future__ import annotations

from haystack import Document, component


@component
class TopKDocuments:
    """Keep the first `top_k` documents after optional score sorting."""

    def __init__(self, top_k: int, sort_by_score: bool = True) -> None:
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if self.sort_by_score and any(document.score is None for document in documents):
            raise ValueError("TopKDocuments requires every document to have a score when sorting.")
        ranked = (
            sorted(
                documents,
                key=lambda document: (float(document.score), document.id),
                reverse=True,
            )
            if self.sort_by_score
            else list(documents)
        )
        return {"documents": ranked[: self.top_k]}
