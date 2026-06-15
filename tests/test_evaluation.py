from pathlib import Path

from omegaconf import OmegaConf

from retrieval_research.io import write_predictions
from retrieval_research.stages.evaluation import _qrels_from_records, run_evaluation


def test_qrels_from_records_groups_positive_judgments() -> None:
    qrels = _qrels_from_records(
        [
            {"query_id": "q1", "document_id": "d1", "relevance": 1},
            {"query_id": "q1", "document_id": "d2", "relevance": 2},
            {"query_id": "q1", "document_id": "d3", "relevance": 0},
            {"query_id": "q2", "document_id": "d4"},
        ]
    )

    assert qrels == {
        "q1": {"d1": 1, "d2": 2},
        "q2": {"d4": 1},
    }


def test_evaluation_reads_prediction_mapping_json(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    qrels_path = tmp_path / "qrels.jsonl"
    metrics_path = tmp_path / "metrics.json"
    output_dir = tmp_path / "run"

    write_predictions(
        predictions_path,
        [
            {
                "query_id": "q1",
                "query": "test query",
                "documents": [{"id": "d1", "content": "doc", "meta": {}, "score": 0.5}],
            }
        ],
    )
    qrels_path.write_text(
        '{"query_id":"q1","document_id":"d1","relevance":1}\n',
        encoding="utf-8",
    )

    cfg = OmegaConf.create(
        {
            "stage": {
                "output_dir": str(output_dir),
                "predictions_path": str(predictions_path),
                "metrics_path": str(metrics_path),
            },
            "dataset": {"qrels_path": str(qrels_path)},
            "metrics": ["Recall@1", "MRR@1"],
        }
    )

    assert run_evaluation(cfg) == {"Recall@1": 1.0, "MRR@1": 1.0}
