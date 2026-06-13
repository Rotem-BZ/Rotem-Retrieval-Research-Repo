"""Top-k and top-p document selectors for retrieval cascades."""

from __future__ import annotations

from haystack import Document, component


def _score(document: Document) -> float:
    return float(document.score or 0.0)


def _rank(documents: list[Document], sort_by_score: bool) -> list[Document]:
    if not sort_by_score:
        return list(documents)
    return sorted(documents, key=lambda document: (_score(document), document.id or ""), reverse=True)


@component
class TopKDocuments:
    """Keep the first `top_k` documents after optional score sorting."""

    def __init__(self, top_k: int, sort_by_score: bool = True) -> None:
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        return {"documents": _rank(documents, self.sort_by_score)[: self.top_k]}


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
        ranked = _rank(documents, self.sort_by_score)
        if self.max_documents is not None:
            ranked = ranked[: self.max_documents]

        positive_scores = [max(_score(document), 0.0) for document in ranked]
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
