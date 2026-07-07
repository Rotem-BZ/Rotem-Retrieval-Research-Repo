"""Weighted score document fusion."""

from __future__ import annotations

from haystack import Document, component

from retrieval_research.utils.documents import copy_document_with_score, sort_documents_by_score


@component
class ScoreFusion:
    """Fuse ranked lists by summing weighted document scores."""

    def __init__(
        self,
        weights: dict[str, float],
        top_k: int | None = None,
        missing_score: float = 0.0,
        normalize_by_source: bool = False,
    ) -> None:
        self.weights = weights
        self.top_k = top_k
        self.missing_score = missing_score
        self.normalize_by_source = normalize_by_source
        component.set_input_types(self, **{source_name: list[Document] for source_name in weights})
        component.set_output_types(self, documents=list[Document])

    def run(self, **ranked_lists: list[Document]) -> dict[str, list[Document]]:
        fused_scores: dict[str, float] = {}
        documents_by_id: dict[str, Document] = {}

        for source_name, weight in self.weights.items():
            documents = ranked_lists.get(source_name, [])
            scores = [
                float(document.score if document.score is not None else self.missing_score)
                for document in documents
            ]
            min_score = min(scores, default=0.0)
            max_score = max(scores, default=0.0)

            for document, raw_score in zip(documents, scores, strict=True):
                if document.id is None:
                    continue
                score = raw_score
                if self.normalize_by_source:
                    score = (
                        1.0
                        if max_score == min_score
                        else (raw_score - min_score) / (max_score - min_score)
                    )
                documents_by_id.setdefault(document.id, document)
                fused_scores[document.id] = fused_scores.get(document.id, 0.0) + (
                    float(weight) * score
                )

        fused = [
            copy_document_with_score(document, fused_scores[document_id])
            for document_id, document in documents_by_id.items()
        ]
        ranked = sort_documents_by_score(fused)
        if self.top_k is not None:
            ranked = ranked[: self.top_k]
        return {"documents": ranked}
