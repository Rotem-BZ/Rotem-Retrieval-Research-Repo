"""Serialization helpers for retrieval prediction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.utils.io.json import read_json, write_json


def predictions_to_mapping(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}

    for prediction in predictions:
        EVALUATION_DATA_SCHEMA.validate_query(prediction)
        query_input = str(prediction[EVALUATION_DATA_SCHEMA.IN])
        payload[query_input] = {
            EVALUATION_DATA_SCHEMA.query_id: str(prediction[EVALUATION_DATA_SCHEMA.query_id]),
            EVALUATION_DATA_SCHEMA.query_content: prediction[EVALUATION_DATA_SCHEMA.query_content],
            "documents": {
                str(document["id"]): {key: value for key, value in document.items() if key != "id"}
                for document in prediction.get("documents", [])
                if document.get("id") is not None
            },
        }

    return payload


def predictions_from_mapping(payload: dict[str, Any]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    for query_input, query_payload in payload.items():
        documents = [
            {"id": document_id, **dict(document_payload)}
            for document_id, document_payload in query_payload.get("documents", {}).items()
        ]
        prediction = {
            EVALUATION_DATA_SCHEMA.query_id: str(query_payload[EVALUATION_DATA_SCHEMA.query_id]),
            EVALUATION_DATA_SCHEMA.IN: str(query_input),
            EVALUATION_DATA_SCHEMA.query_content: query_payload[
                EVALUATION_DATA_SCHEMA.query_content
            ],
            "documents": documents,
        }
        EVALUATION_DATA_SCHEMA.validate_query(prediction)
        predictions.append(prediction)

    return predictions


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    return predictions_from_mapping(read_json(path))


def write_predictions(path: str | Path, predictions: list[dict[str, Any]]) -> Path:
    return write_json(path, predictions_to_mapping(predictions))
