from pathlib import Path

from retrieval_research.io import (
    predictions_from_mapping,
    predictions_to_mapping,
    read_predictions,
    write_predictions,
)


PREDICTIONS = [
    {
        "query_id": "q1",
        "query": "first query",
        "documents": [
            {
                "id": "d1::chunk-0",
                "content": "passage text",
                "meta": {"source_document_id": "d1", "chunk_index": 0},
                "score": 0.9,
            },
            {
                "id": "d2",
                "content": "another passage",
                "meta": {},
                "score": 0.4,
            },
        ],
    }
]


def test_predictions_mapping_uses_query_and_document_ids_as_keys() -> None:
    assert predictions_to_mapping(PREDICTIONS) == {
        "q1": {
            "query": "first query",
            "documents": {
                "d1::chunk-0": {
                    "content": "passage text",
                    "meta": {"source_document_id": "d1", "chunk_index": 0},
                    "score": 0.9,
                },
                "d2": {
                    "content": "another passage",
                    "meta": {},
                    "score": 0.4,
                },
            },
        }
    }


def test_predictions_mapping_round_trips_to_internal_list_shape(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"

    write_predictions(path, PREDICTIONS)

    assert read_predictions(path) == PREDICTIONS
    assert predictions_from_mapping(predictions_to_mapping(PREDICTIONS)) == PREDICTIONS
