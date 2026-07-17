"""Serialization helpers for retrieval prediction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_core.utils.io.json import read_json, write_json


def predictions_to_mapping(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}

    for prediction in predictions:
        query_id = str(prediction["query_id"])
        payload[query_id] = {
            "query": prediction.get("query"),
            "documents": {
                str(document["id"]): {key: value for key, value in document.items() if key != "id"}
                for document in prediction.get("documents", [])
                if document.get("id") is not None
            },
        }

    return payload


def predictions_from_mapping(payload: dict[str, Any]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    for query_id, query_payload in payload.items():
        documents = [
            {"id": document_id, **dict(document_payload)}
            for document_id, document_payload in query_payload.get("documents", {}).items()
        ]
        predictions.append(
            {
                "query_id": query_id,
                "query": query_payload.get("query"),
                "documents": documents,
            }
        )

    return predictions


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    return predictions_from_mapping(read_json(path))


def write_predictions(path: str | Path, predictions: list[dict[str, Any]]) -> Path:
    return write_json(path, predictions_to_mapping(predictions))
