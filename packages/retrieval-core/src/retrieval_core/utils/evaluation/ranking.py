"""TorchMetrics-backed ranking metrics for retrieval experiments."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)


Qrels = dict[str, dict[str, int]]

_METRIC_PATTERN = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9 _-]*)@(?P<k>[1-9][0-9]*)$")
_METRIC_ALIASES = {
    "hitrate": "HitRate",
    "hr": "HitRate",
    "map": "MAP",
    "meanaverageprecision": "MAP",
    "mrr": "MRR",
    "ndcg": "NDCG",
    "normalizeddcg": "NDCG",
    "precision": "Precision",
    "p": "Precision",
    "recall": "Recall",
    "r": "Recall",
}
_METRIC_CLASSES = {
    "HitRate": RetrievalHitRate,
    "MAP": RetrievalMAP,
    "MRR": RetrievalMRR,
    "NDCG": RetrievalNormalizedDCG,
    "Precision": RetrievalPrecision,
    "Recall": RetrievalRecall,
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    k: int

    @property
    def label(self) -> str:
        return f"{self.name}@{self.k}"


def evaluate_rankings(
    predictions: list[dict[str, Any]],
    qrels: Qrels,
    metric_configs: list[str],
) -> dict[str, float]:
    metric_specs = [_parse_metric_spec(metric_config) for metric_config in metric_configs]
    if not metric_specs:
        return {}

    tensors = _build_retrieval_tensors(
        predictions, qrels, max_k=max(spec.k for spec in metric_specs)
    )
    if tensors is None:
        return {spec.label: 0.0 for spec in metric_specs}

    indexes, preds, graded_target = tensors
    binary_target = graded_target > 0
    results: dict[str, float] = {}

    for spec in metric_specs:
        metric_class = _METRIC_CLASSES[spec.name]
        metric = metric_class(top_k=spec.k, empty_target_action="skip")
        target = graded_target if spec.name == "NDCG" else binary_target
        value = metric(preds, target, indexes=indexes)
        results[spec.label] = float(value.item())

    return results


def _parse_metric_spec(metric_config: str) -> MetricSpec:
    if not isinstance(metric_config, str):
        raise TypeError(f"Metric configs must be strings like 'Recall@10', got {metric_config!r}")

    match = _METRIC_PATTERN.match(metric_config.strip())
    if match is None:
        raise ValueError(
            f"Unsupported metric format: {metric_config!r}. Use strings like 'Recall@10'."
        )

    raw_name = re.sub(r"[\s_-]+", "", match.group("name")).lower()
    name = _METRIC_ALIASES.get(raw_name)
    if name is None:
        supported = ", ".join(f"{name}@k" for name in sorted(_METRIC_CLASSES))
        raise ValueError(f"Unsupported metric: {metric_config!r}. Supported metrics: {supported}")

    return MetricSpec(name=name, k=int(match.group("k")))


def _build_retrieval_tensors(
    predictions: list[dict[str, Any]],
    qrels: Qrels,
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    indexes: list[int] = []
    preds: list[float] = []
    target: list[int] = []
    prediction_by_query = {record["query_id"]: record for record in predictions}

    for query_index, query_id in enumerate(qrels):
        judged_documents = qrels[query_id]
        if not judged_documents:
            continue

        retrieved_scores = _collapsed_retrieved_scores(prediction_by_query.get(query_id, {}))
        min_retrieved_score = min(retrieved_scores.values(), default=0.0)
        dummy_score_start = min_retrieved_score - 1.0
        missing_score_start = dummy_score_start - max_k - 1.0

        for document_id, score in retrieved_scores.items():
            indexes.append(query_index)
            preds.append(score)
            target.append(judged_documents.get(document_id, 0))

        for dummy_rank in range(max(0, max_k - len(retrieved_scores))):
            indexes.append(query_index)
            preds.append(dummy_score_start - dummy_rank)
            target.append(0)

        missing_relevance = {
            document_id: relevance
            for document_id, relevance in judged_documents.items()
            if document_id not in retrieved_scores
        }
        for missing_rank, relevance in enumerate(missing_relevance.values()):
            indexes.append(query_index)
            preds.append(missing_score_start - missing_rank)
            target.append(relevance)

    if not indexes:
        return None

    return (
        torch.tensor(indexes, dtype=torch.long),
        torch.tensor(preds, dtype=torch.float),
        torch.tensor(target, dtype=torch.long),
    )


def _collapsed_retrieved_scores(record: dict[str, Any]) -> dict[str, float]:
    retrieved_scores: dict[str, float] = {}

    for rank, document in enumerate(record.get("documents", []), start=1):
        document_id = _evaluation_document_id(document)
        if document_id is None:
            continue

        score = _ranking_score(document, rank)
        previous_score = retrieved_scores.get(document_id)
        if previous_score is None or score > previous_score:
            retrieved_scores[document_id] = score

    return retrieved_scores


def _ranking_score(document: dict[str, Any], rank: int) -> float:
    score = document.get("score")
    if score is None:
        return -float(rank)

    score_value = float(score)
    if not math.isfinite(score_value):
        return -float(rank)

    return score_value - (rank * 1e-12)


def _evaluation_document_id(document: dict[str, Any]) -> str | None:
    meta = document.get("meta") or {}
    return meta.get("source_document_id") or document.get("id")
