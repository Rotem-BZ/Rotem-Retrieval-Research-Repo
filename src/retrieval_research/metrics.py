"""Minimal ranking metrics for retrieval experiments."""

from __future__ import annotations

from typing import Any


def evaluate_rankings(
    predictions: list[dict[str, Any]],
    qrels: dict[str, set[str]],
    metric_configs: list[dict[str, Any]],
) -> dict[str, float]:
    results: dict[str, float] = {}

    for metric_config in metric_configs:
        name = metric_config["name"]
        k = int(metric_config.get("k", 10))

        if name == "recall_at_k":
            results[f"recall@{k}"] = recall_at_k(predictions, qrels, k)
        elif name == "mrr_at_k":
            results[f"mrr@{k}"] = mrr_at_k(predictions, qrels, k)
        else:
            raise ValueError(f"Unsupported metric: {name}")

    return results


def recall_at_k(predictions: list[dict[str, Any]], qrels: dict[str, set[str]], k: int) -> float:
    values: list[float] = []

    for record in predictions:
        query_id = record["query_id"]
        relevant_ids = qrels.get(query_id, set())
        if not relevant_ids:
            continue

        retrieved_ids = _retrieved_ids(record, k)
        values.append(len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids))

    return _mean(values)


def mrr_at_k(predictions: list[dict[str, Any]], qrels: dict[str, set[str]], k: int) -> float:
    values: list[float] = []

    for record in predictions:
        query_id = record["query_id"]
        relevant_ids = qrels.get(query_id, set())
        if not relevant_ids:
            continue

        reciprocal_rank = 0.0
        for rank, document_id in enumerate(_retrieved_ids(record, k), start=1):
            if document_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        values.append(reciprocal_rank)

    return _mean(values)


def _retrieved_ids(record: dict[str, Any], k: int) -> list[str]:
    documents = record.get("documents", [])
    return [document["id"] for document in documents[:k] if document.get("id") is not None]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
