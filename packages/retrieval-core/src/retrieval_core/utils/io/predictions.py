"""Serialization helpers for retrieval prediction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_core.utils.io.json import read_json, write_json


def predictions_to_mapping(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}

    for prediction in predictions:
        query_id = prediction.get("id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("Every prediction requires a non-empty string `id`.")
        if query_id in payload:
            raise ValueError(f"Duplicate prediction query id: {query_id!r}.")
        payload[query_id] = {key: value for key, value in prediction.items() if key != "id"}

    return payload


def predictions_from_mapping(payload: dict[str, Any]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    for query_id, query_payload in payload.items():
        if not isinstance(query_payload, dict):
            raise TypeError(f"Prediction {query_id!r} must be an object.")
        predictions.append({"id": str(query_id), **query_payload})

    return predictions


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    return predictions_from_mapping(read_json(path))


def write_predictions(path: str | Path, predictions: list[dict[str, Any]]) -> Path:
    return write_json(path, predictions_to_mapping(predictions))
