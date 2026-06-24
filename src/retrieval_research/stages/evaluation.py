"""Evaluation stage runner."""

from __future__ import annotations

from omegaconf import DictConfig
from omegaconf import open_dict

from retrieval_research.io import project_path, read_json, read_jsonl, read_predictions, write_json
from retrieval_research.metrics import Qrels, evaluate_rankings
from retrieval_research.pipelines import to_container
from retrieval_research.stages.base import StageContext, is_dry_run


def run_evaluation(cfg: DictConfig) -> dict[str, float]:
    prepare_evaluation_config(cfg)
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
            "predictions_path": str(cfg.stage.predictions_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        },
    )
    return metrics


def prepare_evaluation_config(cfg: DictConfig) -> None:
    run_name = cfg.stage.get("inference_run_name")
    if run_name is None or str(run_name).strip() == "":
        return

    predictions_path = inference_predictions_path_for_run_name(cfg, str(run_name).strip())
    with open_dict(cfg):
        cfg.stage.predictions_path = str(predictions_path)


def inference_predictions_path_for_run_name(cfg: DictConfig, run_name: str):
    runs_dir = project_path(cfg.paths.runs_dir) / "inference"
    matches = _matching_inference_runs(runs_dir, run_name)
    if not matches:
        raise FileNotFoundError(
            f"No inference runs match stage.inference_run_name={run_name!r}. "
            f"Searched under {runs_dir}."
        )
    if len(matches) > 1:
        formatted = "\n".join(f"  - {path.name}" for path in matches)
        raise ValueError(
            f"Multiple inference runs match stage.inference_run_name={run_name!r}:\n"
            f"{formatted}\n"
            "Use a longer prefix or set stage.predictions_path explicitly."
        )

    run_dir = matches[0]
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        predictions_path = result.get("predictions_path")
        if predictions_path:
            return project_path(predictions_path)

    return run_dir / "predictions.json"


def _matching_inference_runs(runs_dir, run_name: str):
    if not runs_dir.exists():
        return []
    return sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and path.name.startswith(run_name)
    )


def _qrels_from_records(records: list[dict]) -> Qrels:
    qrels: Qrels = {}

    for record in records:
        relevance = int(record.get("relevance", 1))
        if relevance <= 0:
            continue

        query_id = record["query_id"]
        document_id = record["document_id"]
        qrels.setdefault(query_id, {})[document_id] = relevance

    return qrels
