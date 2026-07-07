"""JSONL document source."""

from __future__ import annotations

from pathlib import Path

from haystack import Document, component

from retrieval_research.utils.documents import read_jsonl_documents


@component
class JsonlDocumentSource:
    """Read documents from a JSONL dataset file."""

    def __init__(self, documents_path: str) -> None:
        self.documents_path = documents_path

    @component.output_types(documents=list[Document])
    def run(self) -> dict[str, list[Document]]:
        path = Path(self.documents_path)
        if not path.exists():
            raise FileNotFoundError(f"Document dataset not found: {path}")

        return {"documents": read_jsonl_documents(path)}

