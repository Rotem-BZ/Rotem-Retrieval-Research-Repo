"""Evaluation stage runner."""

from __future__ import annotations

import logging

from omegaconf import open_dict

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.stages.base import Stage
from retrieval_core.utils.artifacts import artifact_for_run
from retrieval_core.utils.evaluation import evaluate_rankings
from retrieval_core.utils.io import project_path, read_jsonl, read_predictions, write_json
from retrieval_core.utils.pipelines import to_container

logger = logging.getLogger(__name__)


class EvaluationStage(Stage):
    """Evaluate predictions against the configured relevance judgments."""

    def prepare_config(self) -> None:
        run_id = self.cfg.stage.get("inference_run_id")
        configured_path = self.cfg.stage.get("predictions_path")
        if run_id is None or str(run_id).strip() == "":
            if not configured_path:
                raise ValueError(
                    "Evaluation requires stage.inference_run_id or stage.predictions_path."
                )
            return

        predictions_path = artifact_for_run(
            self.cfg,
            stage_name="inference",
            run_id=str(run_id).strip(),
            artifact_name="predictions",
        )
        if configured_path and project_path(configured_path) != predictions_path:
            raise ValueError(
                "stage.inference_run_id and stage.predictions_path resolve to different artifacts."
            )
        with open_dict(self.cfg):
            self.cfg.stage.predictions_path = str(predictions_path)

    def run(self) -> dict[str, float]:
        predictions = read_predictions(self.cfg.stage.predictions_path)
        qrels: dict[str, dict[str, int]] = {}
        for record in read_jsonl(self.cfg.dataset.qrels_path):
            EVALUATION_DATA_SCHEMA.validate_qrel(record)
            label = int(record[EVALUATION_DATA_SCHEMA.label])
            if label > 0:
                query_input = str(record[EVALUATION_DATA_SCHEMA.IN])
                document_id = str(record[EVALUATION_DATA_SCHEMA.doc_id])
                qrels.setdefault(query_input, {})[document_id] = label
        logger.info(
            "Evaluating predictions: predictions=%d judged_queries=%d metrics=%s",
            len(predictions),
            len(qrels),
            list(self.cfg.metrics),
        )
        metrics = evaluate_rankings(predictions, qrels, to_container(self.cfg.metrics))

        metrics_path = write_json(self.cfg.stage.metrics_path, metrics)

        self.write_result(
            {
                "predictions_path": str(self.cfg.stage.predictions_path),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            },
        )
        inputs = {"predictions_path": str(project_path(self.cfg.stage.predictions_path))}
        if self.cfg.stage.get("inference_run_id"):
            inputs["inference_run_id"] = str(self.cfg.stage.inference_run_id)
        self.write_manifest(artifacts={"metrics": metrics_path}, inputs=inputs)
        logger.info("Metrics written: path=%s metrics=%s", metrics_path, metrics)
        return metrics
