"""Evaluation stage runner."""

from __future__ import annotations

from omegaconf import DictConfig
from omegaconf import open_dict

from retrieval_core.stages.base import StageContext
from retrieval_core.utils.artifacts import artifact_for_run
from retrieval_core.utils.evaluation import evaluate_rankings
from retrieval_core.utils.io import project_path, read_jsonl, read_predictions, write_json
from retrieval_core.utils.pipelines import to_container


def run_evaluation(cfg: DictConfig) -> dict[str, float]:
    prepare_evaluation_config(cfg)
    context = StageContext.from_config(cfg)
    predictions = read_predictions(cfg.stage.predictions_path)
    qrels: dict[str, dict[str, int]] = {}
    for record in read_jsonl(cfg.dataset.qrels_path):
        relevance = int(record.get("relevance", 1))
        if relevance > 0:
            qrels.setdefault(record["query_id"], {})[record["document_id"]] = relevance
    metrics = evaluate_rankings(predictions, qrels, to_container(cfg.metrics))

    metrics_path = write_json(cfg.stage.metrics_path, metrics)

    context.write_resolved_config()
    context.write_result(
        {
            "predictions_path": str(cfg.stage.predictions_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        },
    )
    inputs = {"predictions_path": str(project_path(cfg.stage.predictions_path))}
    if cfg.stage.get("inference_run_id"):
        inputs["inference_run_id"] = str(cfg.stage.inference_run_id)
    context.write_manifest(artifacts={"metrics": metrics_path}, inputs=inputs)
    return metrics


def prepare_evaluation_config(cfg: DictConfig) -> None:
    run_id = cfg.stage.get("inference_run_id")
    configured_path = cfg.stage.get("predictions_path")
    if run_id is None or str(run_id).strip() == "":
        if not configured_path:
            raise ValueError(
                "Evaluation requires stage.inference_run_id or stage.predictions_path."
            )
        return

    predictions_path = artifact_for_run(
        cfg,
        stage_name="inference",
        run_id=str(run_id).strip(),
        artifact_name="predictions",
    )
    if configured_path and project_path(configured_path) != predictions_path:
        raise ValueError(
            "stage.inference_run_id and stage.predictions_path resolve to different artifacts."
        )
    with open_dict(cfg):
        cfg.stage.predictions_path = str(predictions_path)
