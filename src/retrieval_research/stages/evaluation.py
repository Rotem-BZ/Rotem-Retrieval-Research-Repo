"""Evaluation stage runner."""

from __future__ import annotations

from omegaconf import DictConfig

from retrieval_research.io import read_jsonl, write_json
from retrieval_research.metrics import evaluate_rankings
from retrieval_research.pipelines import to_container
from retrieval_research.stages.base import StageContext


def run_evaluation(cfg: DictConfig) -> dict[str, float]:
    context = StageContext.from_config(cfg)
    predictions = read_jsonl(cfg.stage.predictions_path)
    qrels = _qrels_from_dataset(read_jsonl(cfg.dataset.queries_path))
    metrics = evaluate_rankings(predictions, qrels, to_container(cfg.metrics))

    metrics_path = write_json(cfg.stage.metrics_path, metrics)
    context.write_resolved_config()
    context.write_result(
        {
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        },
    )
    return metrics


def _qrels_from_dataset(queries: list[dict]) -> dict[str, set[str]]:
    return {
        query["id"]: set(query.get("relevant_document_ids", []))
        for query in queries
    }
