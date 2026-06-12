from retrieval_research.stages.evaluation import _qrels_from_records


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
        "q1": {"d1", "d2"},
        "q2": {"d4"},
    }
