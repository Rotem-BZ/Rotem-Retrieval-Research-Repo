"""Weighted score fusion with fixed per-source normalization strategies."""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean, pstdev

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


def _linear_normalize(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _z_normalize(scores: list[float]) -> list[float]:
    mean = fmean(scores)
    standard_deviation = pstdev(scores)
    if standard_deviation == 0:
        return [0.0] * len(scores)
    return [(score - mean) / standard_deviation for score in scores]


def _fuse_scores(
    *,
    ranked_lists: dict[str, list[Document]],
    weights: dict[str, float],
    top_k: int | None,
    missing_score: float,
    normalize: Callable[[list[float]], list[float]],
) -> dict[str, list[Document]]:
    fused_scores: dict[str, float] = {}
    documents_by_id: dict[str, Document] = {}

    for source_name, weight in weights.items():
        documents = ranked_lists.get(source_name, [])
        scores = [
            float(document.score if document.score is not None else missing_score)
            for document in documents
        ]
        normalized_scores = normalize(scores) if scores else scores

        for document, score in zip(documents, normalized_scores, strict=True):
            if document.id is None:
                continue
            documents_by_id.setdefault(document.id, document)
            fused_scores[document.id] = fused_scores.get(document.id, 0.0) + (float(weight) * score)

    fused = [
        _copy_document_with_score(document, fused_scores[document_id])
        for document_id, document in documents_by_id.items()
    ]
    ranked = _sort_documents_by_score(fused)
    if top_k is not None:
        ranked = ranked[:top_k]
    return {"documents": ranked}


@component
class LinearScoreFusion:
    """Fuse weighted scores after min-max normalization within each source."""

    def __init__(
        self,
        weights: dict[str, float],
        top_k: int | None = None,
        missing_score: float = 0.0,
    ) -> None:
        self.weights = weights
        self.top_k = top_k
        self.missing_score = missing_score
        component.set_input_types(self, **{source_name: list[Document] for source_name in weights})
        component.set_output_types(self, documents=list[Document])

    def run(self, **ranked_lists: list[Document]) -> dict[str, list[Document]]:
        return _fuse_scores(
            ranked_lists=ranked_lists,
            weights=self.weights,
            top_k=self.top_k,
            missing_score=self.missing_score,
            normalize=_linear_normalize,
        )


@component
class ZScoreFusion:
    """Fuse weighted scores after Z-normalization within each source."""

    def __init__(
        self,
        weights: dict[str, float],
        top_k: int | None = None,
        missing_score: float = 0.0,
    ) -> None:
        self.weights = weights
        self.top_k = top_k
        self.missing_score = missing_score
        component.set_input_types(self, **{source_name: list[Document] for source_name in weights})
        component.set_output_types(self, documents=list[Document])

    def run(self, **ranked_lists: list[Document]) -> dict[str, list[Document]]:
        return _fuse_scores(
            ranked_lists=ranked_lists,
            weights=self.weights,
            top_k=self.top_k,
            missing_score=self.missing_score,
            normalize=_z_normalize,
        )
