"""Top-k document selector."""

from __future__ import annotations

from haystack import Document, component


@component
class TopKDocuments:
    """Keep the `top_k` highest-scoring documents."""

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if any(document.score is None for document in documents):
            raise ValueError("TopKDocuments requires every document to have a score.")
        ranked = sorted(
            documents,
            key=lambda document: (float(document.score), document.id),
            reverse=True,
        )
        return {"documents": ranked[: self.top_k]}
