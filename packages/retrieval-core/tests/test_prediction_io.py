from pathlib import Path

import pytest

from retrieval_core.utils.io import (
    predictions_from_mapping,
    predictions_to_mapping,
    read_predictions,
    write_predictions,
)

PREDICTIONS = [
    {
        "id": "q1",
        "data": {"final_query": "reformulated query"},
        "results": [
            {
                "id": "d1::chunk-0",
                "document_id": "d1",
                "score": 0.9,
                "data": {"chunk_index": 0},
            },
            {"id": "d2", "document_id": "d2", "score": 0.4},
        ],
    }
]


def test_predictions_mapping_keeps_ordered_results_and_extra_data() -> None:
    assert predictions_to_mapping(PREDICTIONS) == {
        "q1": {
            "data": {"final_query": "reformulated query"},
            "results": [
                {
                    "id": "d1::chunk-0",
                    "document_id": "d1",
                    "score": 0.9,
                    "data": {"chunk_index": 0},
                },
                {"id": "d2", "document_id": "d2", "score": 0.4},
            ],
        }
    }


def test_predictions_mapping_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    write_predictions(path, PREDICTIONS)
    assert read_predictions(path) == PREDICTIONS
    assert predictions_from_mapping(predictions_to_mapping(PREDICTIONS)) == PREDICTIONS


def test_duplicate_query_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate prediction query id"):
        predictions_to_mapping([{"id": "q1", "results": []}, {"id": "q1", "results": []}])
