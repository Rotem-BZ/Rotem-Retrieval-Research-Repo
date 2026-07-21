"""Canonical index artifact paths and discovery."""

from __future__ import annotations

from pathlib import Path

from retrieval_core.utils.io import project_path

INDEX_FILENAME = "index.jsonl"
INDEX_ID_FORBIDDEN_CHARS = {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}


def validate_index_id(index_id: str) -> str:
    """Return a normalized index id that is safe as one directory name."""

    normalized = str(index_id).strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or Path(normalized).name != normalized
        or any(character in normalized for character in INDEX_ID_FORBIDDEN_CHARS)
    ):
        raise ValueError(f"Index id must be one directory name, got {index_id!r}.")
    return normalized


def index_artifact_path(indexes_dir: str | Path, index_id: str) -> Path:
    """Return the canonical JSONL artifact path for an index id."""

    return project_path(indexes_dir) / validate_index_id(index_id) / INDEX_FILENAME


def discover_index_ids(indexes_dir: str | Path) -> list[str]:
    """Return valid index ids that contain a canonical index artifact."""

    root = project_path(indexes_dir)
    if not root.is_dir():
        return []

    index_ids: list[str] = []
    for directory in root.iterdir():
        if not directory.is_dir() or not (directory / INDEX_FILENAME).is_file():
            continue
        try:
            index_ids.append(validate_index_id(directory.name))
        except ValueError:
            continue
    return sorted(index_ids)
