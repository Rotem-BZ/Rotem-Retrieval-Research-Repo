from pathlib import Path

import pytest
from omegaconf import OmegaConf

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.stages.evaluation import prepare_evaluation_config, run_evaluation
from retrieval_core.utils.artifacts import discover_inference_run_ids
from retrieval_core.utils.io import write_json, write_jsonl, write_predictions


def test_evaluation_reads_prediction_mapping_json(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    qrels_path = tmp_path / "qrels.jsonl"
    metrics_path = tmp_path / "metrics.json"
    output_dir = tmp_path / "run"

    write_predictions(
        predictions_path,
        [
            {
                EVALUATION_DATA_SCHEMA.query_id: "external-q1",
                EVALUATION_DATA_SCHEMA.IN: "q1",
                EVALUATION_DATA_SCHEMA.query_content: "test query",
                "documents": [{"id": "d1", "content": "doc", "meta": {}, "score": 0.5}],
            }
        ],
    )
    write_jsonl(
        qrels_path,
        [
            {
                EVALUATION_DATA_SCHEMA.IN: "q1",
                EVALUATION_DATA_SCHEMA.doc_id: "d1",
                EVALUATION_DATA_SCHEMA.label: 1,
                "annotation_source": "test",
            }
        ],
    )

    cfg = OmegaConf.create(
        {
            "stage": {
                "name": "evaluation",
                "run_id": "evaluation_1",
                "output_dir": str(output_dir),
                "predictions_path": str(predictions_path),
                "metrics_path": str(metrics_path),
                "inference_run_id": None,
            },
            "paths": {"runs_dir": str(tmp_path / "runs")},
            "dataset": {"qrels_path": str(qrels_path)},
            "metrics": ["Recall@1", "MRR@1"],
        }
    )

    assert run_evaluation(cfg) == {"Recall@1": 1.0, "MRR@1": 1.0}


def test_evaluation_resolves_prediction_path_from_exact_inference_run_id(tmp_path: Path) -> None:
    predictions_path = tmp_path / "runs" / "inference" / "bge_20260101_010101" / "predictions.json"
    write_predictions(
        predictions_path,
        [
            {
                EVALUATION_DATA_SCHEMA.query_id: "external-q1",
                EVALUATION_DATA_SCHEMA.IN: "q1",
                EVALUATION_DATA_SCHEMA.query_content: "test query",
                "documents": [{"id": "d1", "content": "doc", "meta": {}, "score": 0.5}],
            }
        ],
    )
    write_json(
        predictions_path.parent / "manifest.json",
        {"artifacts": {"predictions": str(predictions_path)}},
    )

    cfg = OmegaConf.create(
        {
            "paths": {"runs_dir": str(tmp_path / "runs")},
            "stage": {"inference_run_id": "bge_20260101_010101"},
        }
    )

    prepare_evaluation_config(cfg)

    assert Path(cfg.stage.predictions_path) == predictions_path


def test_evaluation_does_not_accept_inference_run_prefixes(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs" / "inference"
    runs_dir.joinpath("bge_20260101_010101").mkdir(parents=True)
    runs_dir.joinpath("bge_20260101_020202").mkdir(parents=True)
    cfg = OmegaConf.create(
        {
            "paths": {"runs_dir": str(tmp_path / "runs")},
            "stage": {"inference_run_id": "bge"},
        }
    )

    with pytest.raises(FileNotFoundError, match="No inference run exists"):
        prepare_evaluation_config(cfg)


def test_discovers_completed_inference_runs_for_selected_dataset(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_inference_run(runs_dir, "toy-current", dataset_name="toy")
    _write_inference_run(runs_dir, "other", dataset_name="other")
    _write_inference_run(runs_dir, "toy-legacy", dataset_name="toy", legacy=True)
    incomplete = runs_dir / "inference" / "incomplete"
    incomplete.mkdir(parents=True)
    write_json(incomplete / "manifest.json", {"inputs": {"dataset": "toy"}})

    assert discover_inference_run_ids(runs_dir, dataset_name="toy") == [
        "toy-current",
        "toy-legacy",
    ]


def _write_inference_run(
    runs_dir: Path,
    run_id: str,
    *,
    dataset_name: str,
    legacy: bool = False,
) -> None:
    run_dir = runs_dir / "inference" / run_id
    predictions_path = run_dir / "predictions.json"
    write_predictions(predictions_path, [])
    manifest = {
        "stage": {"name": "inference", "run_id": run_id},
        "artifacts": {"predictions": str(predictions_path)},
        "inputs": {} if legacy else {"dataset": dataset_name},
    }
    write_json(run_dir / "manifest.json", manifest)
    if legacy:
        (run_dir / "resolved_config.yaml").write_text(
            f"dataset:\n  name: {dataset_name}\n",
            encoding="utf-8",
        )
