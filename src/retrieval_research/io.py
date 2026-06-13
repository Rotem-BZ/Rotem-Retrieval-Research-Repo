"""Small IO utilities shared by stage runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf


def project_path(path: str | Path) -> Path:
    """Resolve a path relative to the original working directory Hydra saw."""

    candidate = Path(str(path))
    if candidate.is_absolute():
        return candidate

    try:
        base = Path(get_original_cwd())
    except ValueError:
        base = Path.cwd()
    return base / candidate


def ensure_parent(path: str | Path) -> Path:
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    resolved = project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = project_path(path)
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_json(path: str | Path) -> Any:
    resolved = project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return resolved


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
    return resolved


def write_text(path: str | Path, text: str) -> Path:
    resolved = ensure_parent(path)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def predictions_to_mapping(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}

    for prediction in predictions:
        query_id = str(prediction["query_id"])
        payload[query_id] = {
            "query": prediction.get("query"),
            "documents": {
                str(document["id"]): {
                    key: value for key, value in document.items() if key != "id"
                }
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


def config_to_yaml(config: DictConfig) -> str:
    return OmegaConf.to_yaml(config, resolve=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return to_jsonable(OmegaConf.to_container(value, resolve=True))

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value
