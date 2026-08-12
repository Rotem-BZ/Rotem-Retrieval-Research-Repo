import pytest

from retrieval_core.utils.evaluation import evaluate_rankings

PREDICTIONS = [
    {
        "id": "query-1",
        "results": [
            {"id": "d1", "document_id": "d1", "score": 2.0},
            {"id": "d2", "document_id": "d2", "score": 1.0},
        ],
    },
    {
        "id": "query-2",
        "results": [
            {"id": "d4", "document_id": "d4", "score": 2.0},
            {"id": "d3", "document_id": "d3", "score": 1.0},
        ],
    },
]
QRELS = {"q1": {"d1": 1}, "q2": {"d3": 1}}
QUERY_INPUTS = {"query-1": "q1", "query-2": "q2"}


def test_evaluate_rankings_with_torchmetrics() -> None:
    metrics = evaluate_rankings(
        PREDICTIONS,
        QRELS,
        ["Recall@2", "MRR@2", "Precision@2", "HitRate@2", "NDCG@2"],
        QUERY_INPUTS,
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


def test_metrics_collapse_chunks_by_source_document() -> None:
    predictions = [
        {
            "id": "query-1",
            "results": [
                {"id": "d2::chunk-0", "document_id": "d2", "score": 3.0},
                {"id": "d1::chunk-0", "document_id": "d1", "score": 2.0},
                {"id": "d1::chunk-1", "document_id": "d1", "score": 1.0},
            ],
        }
    ]
    metrics = evaluate_rankings(
        predictions, {"q1": {"d1": 1}}, ["Recall@3", "MRR@3"], {"query-1": "q1"}
    )
    assert metrics == pytest.approx({"Recall@3": 1.0, "MRR@3": 0.5})


def test_missing_relevant_documents_do_not_become_retrieved() -> None:
    metrics = evaluate_rankings(
        [{"id": "query-1", "results": [{"id": "d2", "document_id": "d2", "score": 1.0}]}],
        {"q1": {"d1": 1}},
        ["Recall@10", "Precision@10", "HitRate@10"],
        {"query-1": "q1"},
    )
    assert metrics == pytest.approx(
        {"Recall@10": 0.0, "Precision@10": 0.0, "HitRate@10": 0.0}
    )


def test_metric_strings_are_normalized() -> None:
    metrics = evaluate_rankings(
        PREDICTIONS, QRELS, ["recall@2", "hit_rate@2", "p@2"], QUERY_INPUTS
    )
    assert set(metrics) == {"Recall@2", "HitRate@2", "Precision@2"}


def test_metric_strings_must_use_metric_at_k_format() -> None:
    with pytest.raises(ValueError, match="Recall@10"):
        evaluate_rankings(PREDICTIONS, QRELS, ["recall_at_k"], QUERY_INPUTS)


def test_queries_sharing_an_information_need_are_scored_independently() -> None:
    predictions = [
        {"id": "q1-a", "results": [{"id": "d1", "document_id": "d1", "score": 1.0}]},
        {"id": "q1-b", "results": [{"id": "d2", "document_id": "d2", "score": 1.0}]},
    ]
    metrics = evaluate_rankings(
        predictions,
        {"need-1": {"d1": 1}},
        ["Recall@1"],
        {"q1-a": "need-1", "q1-b": "need-1"},
    )
    assert metrics == pytest.approx({"Recall@1": 0.5})


def test_predictions_must_reference_dataset_queries() -> None:
    with pytest.raises(ValueError, match="unknown query id"):
        evaluate_rankings([{"id": "missing", "results": []}], QRELS, ["Recall@1"], {})
