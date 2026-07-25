"""JSONL document indexer."""

from __future__ import annotations

import json
from pathlib import Path

from haystack import Document, component


@component
class JsonlDocumentIndexer:
    """Write Haystack documents to a JSONL index artifact."""

    def __init__(self, output_path: str, overwrite: bool = True) -> None:
        self.output_path = output_path
        self.overwrite = overwrite
        self._batch_path: Path | None = None

    def begin_batch_write(self, output_path: str) -> None:
        """Start an incremental write session at a stage-owned temporary path."""

        if self._batch_path is not None:
            raise RuntimeError("A JSONL batch write session is already active.")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Temporary index path already exists: {path}")
        path.touch()
        self._batch_path = path

    def finish_batch_write(self) -> None:
        """Finish the active session without publishing its temporary artifact."""

        if self._batch_path is None:
            raise RuntimeError("No JSONL batch write session is active.")
        self._batch_path = None

    def abort_batch_write(self) -> None:
        """Discard in-memory session state; the stage owns temporary-file cleanup."""

        self._batch_path = None

    @component.output_types(index_path=str, indexed_count=int)
    def run(self, documents: list[Document]) -> dict[str, str | int]:
        path = self._batch_path or Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._batch_path is None and path.exists() and not self.overwrite:
            raise FileExistsError(f"Index already exists and overwrite=false: {path}")

        mode = "a" if self._batch_path is not None else "w"
        with path.open(mode, encoding="utf-8") as handle:
            for document in documents:
                record = {
                    "id": document.id,
                    "content": document.content,
                    "meta": dict(document.meta or {}),
                    "score": getattr(document, "score", None),
                }
                embedding = getattr(document, "embedding", None)
                if embedding is not None:
                    if hasattr(embedding, "tolist") and callable(embedding.tolist):
                        embedding = embedding.tolist()
                    record["embedding"] = embedding
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {"index_path": str(path), "indexed_count": len(documents)}
