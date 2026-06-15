import pytest

from retrieval_research.metrics import evaluate_rankings


PREDICTIONS = [
    {
        "query_id": "q1",
        "documents": [
            {"id": "d1", "score": 2.0},
            {"id": "d2", "score": 1.0},
        ],
    },
    {
        "query_id": "q2",
        "documents": [
            {"id": "d4", "score": 2.0},
            {"id": "d3", "score": 1.0},
        ],
    },
]

QRELS = {
    "q1": {"d1": 1},
    "q2": {"d3": 1},
}


def test_evaluate_rankings_with_torchmetrics() -> None:
    metrics = evaluate_rankings(
        PREDICTIONS,
        QRELS,
        [
            "Recall@2",
            "MRR@2",
            "Precision@2",
            "HitRate@2",
            "NDCG@2",
        ],
    )

    assert metrics == pytest.approx(
        {
            "Recall@2": 1.0,
            "MRR@2": 0.75,
            "Precision@2": 0.5,
            "HitRate@2": 1.0,
            "NDCG@2": 0.8154648767857288,
        }
    )


def test_metrics_use_source_document_id_for_chunk_predictions() -> None:
    predictions = [
        {
            "query_id": "q1",
            "documents": [
                {"id": "d2::chunk-0", "meta": {"source_document_id": "d2"}, "score": 3.0},
                {"id": "d1::chunk-0", "meta": {"source_document_id": "d1"}, "score": 2.0},
                {"id": "d1::chunk-1", "meta": {"source_document_id": "d1"}, "score": 1.0},
            ],
        }
    ]

    metrics = evaluate_rankings(predictions, {"q1": {"d1": 1}}, ["Recall@3", "MRR@3"])

    assert metrics == pytest.approx(
        {
            "Recall@3": 1.0,
            "MRR@3": 0.5,
        }
    )


def test_missing_relevant_documents_do_not_become_retrieved() -> None:
    metrics = evaluate_rankings(
        [{"query_id": "q1", "documents": [{"id": "d2", "score": 1.0}]}],
        {"q1": {"d1": 1}},
        ["Recall@10", "Precision@10", "HitRate@10"],
    )

    assert metrics == pytest.approx(
        {
            "Recall@10": 0.0,
            "Precision@10": 0.0,
            "HitRate@10": 0.0,
        }
    )


def test_metric_strings_are_normalized() -> None:
    metrics = evaluate_rankings(PREDICTIONS, QRELS, ["recall@2", "hit_rate@2", "p@2"])

    assert set(metrics) == {"Recall@2", "HitRate@2", "Precision@2"}


def test_metric_strings_must_use_metric_at_k_format() -> None:
    with pytest.raises(ValueError, match="Recall@10"):
        evaluate_rankings(PREDICTIONS, QRELS, ["recall_at_k"])
