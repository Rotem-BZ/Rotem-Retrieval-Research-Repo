from retrieval_research.metrics import evaluate_rankings, mrr_at_k, recall_at_k


PREDICTIONS = [
    {
        "query_id": "q1",
        "documents": [
            {"id": "d1"},
            {"id": "d2"},
        ],
    },
    {
        "query_id": "q2",
        "documents": [
            {"id": "d4"},
            {"id": "d3"},
        ],
    },
]

QRELS = {
    "q1": {"d1"},
    "q2": {"d3"},
}


def test_recall_at_k() -> None:
    assert recall_at_k(PREDICTIONS, QRELS, 1) == 0.5
    assert recall_at_k(PREDICTIONS, QRELS, 2) == 1.0


def test_mrr_at_k() -> None:
    assert mrr_at_k(PREDICTIONS, QRELS, 2) == 0.75


def test_evaluate_rankings() -> None:
    metrics = evaluate_rankings(
        PREDICTIONS,
        QRELS,
        [
            {"name": "recall_at_k", "k": 2},
            {"name": "mrr_at_k", "k": 2},
        ],
    )

    assert metrics == {
        "recall@2": 1.0,
        "mrr@2": 0.75,
    }


def test_metrics_use_source_document_id_for_chunk_predictions() -> None:
    predictions = [
        {
            "query_id": "q1",
            "documents": [
                {"id": "d2::chunk-0", "meta": {"source_document_id": "d2"}},
                {"id": "d1::chunk-0", "meta": {"source_document_id": "d1"}},
                {"id": "d1::chunk-1", "meta": {"source_document_id": "d1"}},
            ],
        }
    ]

    assert recall_at_k(predictions, {"q1": {"d1"}}, 3) == 1.0
    assert mrr_at_k(predictions, {"q1": {"d1"}}, 3) == 0.5
