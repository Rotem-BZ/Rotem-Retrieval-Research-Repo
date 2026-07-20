"""Reciprocal-rank document fusion."""

from __future__ import annotations

from haystack import Document, component


def _copy_document_with_score(document: Document, score: float) -> Document:
    return Document(
        id=document.id,
        content=document.content,
        meta=dict(document.meta or {}),
        score=score,
        embedding=getattr(document, "embedding", None),
    )


def _sort_documents_by_score(documents: list[Document]) -> list[Document]:
    return sorted(
        documents,
        key=lambda document: (float(document.score or 0.0), document.id or ""),
        reverse=True,
    )


@component
class ReciprocalRankFusion:
    """Fuse ranked lists with weighted reciprocal rank fusion."""

    def __init__(
        self,
        weights: dict[str, float],
        top_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.weights = weights
        self.top_k = top_k
        self.rrf_k = rrf_k
        component.set_input_types(self, **{source_name: list[Document] for source_name in weights})
        component.set_output_types(self, documents=list[Document])

    def run(self, **ranked_lists: list[Document]) -> dict[str, list[Document]]:
        fused_scores: dict[str, float] = {}
        documents_by_id: dict[str, Document] = {}

        for source_name, weight in self.weights.items():
            for rank, document in enumerate(ranked_lists.get(source_name, []), start=1):
                if document.id is None:
                    continue
                documents_by_id.setdefault(document.id, document)
                fused_scores[document.id] = fused_scores.get(document.id, 0.0) + (
                    float(weight) / (self.rrf_k + rank)
                )

        fused = [
            _copy_document_with_score(document, fused_scores[document_id])
            for document_id, document in documents_by_id.items()
        ]
        ranked = _sort_documents_by_score(fused)
        if self.top_k is not None:
            ranked = ranked[: self.top_k]
        return {"documents": ranked}
