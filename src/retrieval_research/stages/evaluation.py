"""Evaluation stage runner."""

from __future__ import annotations

from omegaconf import DictConfig

from retrieval_research.io import read_jsonl, read_predictions, write_json
from retrieval_research.metrics import evaluate_rankings
from retrieval_research.pipelines import to_container
from retrieval_research.stages.base import StageContext, is_dry_run


def run_evaluation(cfg: DictConfig) -> dict[str, float]:
    context = StageContext.from_config(cfg)
    predictions = read_predictions(cfg.stage.predictions_path)
    qrels = _qrels_from_records(read_jsonl(cfg.dataset.qrels_path))
    metrics = evaluate_rankings(predictions, qrels, to_container(cfg.metrics))

    if is_dry_run(cfg):
        metrics_path = cfg.stage.metrics_path
    else:
        metrics_path = write_json(cfg.stage.metrics_path, metrics)

    context.write_resolved_config()
    context.write_result(
        {
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        },
    )
    return metrics


def _qrels_from_records(records: list[dict]) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}

    for record in records:
        relevance = int(record.get("relevance", 1))
        if relevance <= 0:
            continue

        query_id = record["query_id"]
        document_id = record["document_id"]
        qrels.setdefault(query_id, set()).add(document_id)

    return qrels
