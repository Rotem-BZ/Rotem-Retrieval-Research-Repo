"""JSON and JSON Lines readers and writers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from retrieval_core.utils.io.paths import ensure_parent, project_path
from retrieval_core.utils.io.serialization import to_jsonable


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


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Write JSON through a sibling temporary file and atomically replace the target."""

    resolved = ensure_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        dir=resolved.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary_path.replace(resolved)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return resolved


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
    return resolved
