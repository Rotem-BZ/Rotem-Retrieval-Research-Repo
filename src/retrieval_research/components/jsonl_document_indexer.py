"""JSONL document indexer."""

from __future__ import annotations

import json
from pathlib import Path

from haystack import Document, component

from retrieval_research.components.dummy_utils import document_to_record


@component
class JsonlDocumentIndexer:
    """Write Haystack documents to a JSONL index artifact."""

    def __init__(self, output_path: str, overwrite: bool = True) -> None:
        self.output_path = output_path
        self.overwrite = overwrite

    @component.output_types(index_path=str, indexed_count=int)
    def run(self, documents: list[Document]) -> dict[str, str | int]:
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not self.overwrite:
            raise FileExistsError(f"Index already exists and overwrite=false: {path}")

        with path.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document_to_record(document), ensure_ascii=False) + "\n")

        return {"index_path": str(path), "indexed_count": len(documents)}
