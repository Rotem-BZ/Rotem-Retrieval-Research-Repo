"""JSONL document source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haystack import Document, component


def _document_from_record(
    record: dict[str, Any],
    *,
    id_field: str,
    content_field: str,
) -> Document:
    missing_fields = [field for field in (id_field, content_field) if field not in record]
    if missing_fields:
        raise ValueError(f"Document record is missing required fields: {missing_fields}")
    reserved_fields = {id_field, content_field, "meta", "score", "embedding"}
    meta = {key: value for key, value in record.items() if key not in reserved_fields}
    meta.update(dict(record.get("meta") or {}))
    return Document(
        id=str(record[id_field]),
        content=str(record[content_field]),
        meta=meta,
        score=record.get("score"),
        embedding=record.get("embedding"),
    )


def _read_jsonl_documents(
    path: Path,
    *,
    id_field: str,
    content_field: str,
) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(
                    _document_from_record(
                        json.loads(line),
                        id_field=id_field,
                        content_field=content_field,
                    )
                )
    return documents


@component
class JsonlDocumentSource:
    """Read documents from a JSONL dataset file."""

    def __init__(self, documents_path: str, id_field: str, content_field: str) -> None:
        self.documents_path = documents_path
        self.id_field = id_field
        self.content_field = content_field

    @component.output_types(documents=list[Document])
    def run(self) -> dict[str, list[Document]]:
        path = Path(self.documents_path)
        if not path.exists():
            raise FileNotFoundError(f"Document dataset not found: {path}")

        return {
            "documents": _read_jsonl_documents(
                path,
                id_field=self.id_field,
                content_field=self.content_field,
            )
        }
