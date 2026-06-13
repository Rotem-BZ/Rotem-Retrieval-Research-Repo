"""Legacy weighted reciprocal-rank fusion component."""

from __future__ import annotations

from haystack import Document, component


@component
class WeightedDocumentFusion:
    """Fuse ranked document lists with weighted reciprocal rank fusion."""

    def __init__(
        self,
        weights: dict[str, float],
        top_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.weights = weights
        self.top_k = top_k
        self.rrf_k = rrf_k

        component.set_input_types(
            self,
            **{source_name: list[Document] for source_name in weights},
        )
        component.set_output_types(self, documents=list[Document])

    def run(self, **ranked_lists: list[Document]) -> dict[str, list[Document]]:
        fused_scores: dict[str, float] = {}
        documents_by_id: dict[str, Document] = {}

        for source_name, weight in self.weights.items():
            documents = ranked_lists.get(source_name, [])
            for rank, document in enumerate(documents, start=1):
                document_id = document.id
                if document_id is None:
                    continue

                documents_by_id.setdefault(document_id, document)
                fused_scores[document_id] = fused_scores.get(document_id, 0.0) + (
                    float(weight) / (self.rrf_k + rank)
                )

        fused = [
            Document(
                id=document.id,
                content=document.content,
                meta=dict(document.meta or {}),
                score=fused_scores[document_id],
            )
            for document_id, document in documents_by_id.items()
        ]
        fused.sort(key=lambda document: (document.score or 0.0, document.id or ""), reverse=True)

        if self.top_k is not None:
            fused = fused[: self.top_k]

        return {"documents": fused}
