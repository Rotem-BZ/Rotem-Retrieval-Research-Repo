"""Top-p document selector."""

from __future__ import annotations

from haystack import Document, component

from retrieval_research.utils.documents import document_score, sort_documents_by_score


@component
class TopPDocuments:
    """Keep documents until cumulative positive score mass reaches `top_p`."""

    def __init__(
        self,
        top_p: float,
        min_documents: int = 1,
        max_documents: int | None = None,
        sort_by_score: bool = True,
    ) -> None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1].")
        self.top_p = top_p
        self.min_documents = min_documents
        self.max_documents = max_documents
        self.sort_by_score = sort_by_score

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        ranked = sort_documents_by_score(documents) if self.sort_by_score else list(documents)
        if self.max_documents is not None:
            ranked = ranked[: self.max_documents]

        positive_scores = [max(document_score(document), 0.0) for document in ranked]
        total = sum(positive_scores)
        if total <= 0:
            return {"documents": ranked[: self.min_documents]}

        selected: list[Document] = []
        cumulative = 0.0

        for document, positive_score in zip(ranked, positive_scores, strict=True):
            selected.append(document)
            cumulative += positive_score
            if len(selected) >= self.min_documents and cumulative / total >= self.top_p:
                break

        return {"documents": selected}
