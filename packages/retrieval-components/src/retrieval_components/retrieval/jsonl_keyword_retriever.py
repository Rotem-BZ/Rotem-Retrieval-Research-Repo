"""JSONL keyword retriever."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from haystack import Document, component


def _document_from_record(record: dict[str, Any]) -> Document:
    return Document(
        id=record.get("id"),
        content=record.get("content", ""),
        meta=dict(record.get("meta") or {}),
        score=record.get("score"),
        embedding=record.get("embedding"),
    )


def _read_jsonl_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(_document_from_record(json.loads(line)))
    return documents


def _candidate_document_id(document: Document) -> str | None:
    meta = document.meta or {}
    return meta.get("source_document_id") or document.id


@component
class JsonlKeywordRetriever:
    """A tiny lexical retriever over the JSONL artifact written by the dummy indexer."""

    def __init__(self, index_path: str, top_k: int = 5) -> None:
        self.index_path = index_path
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        top_k: int | None = None,
        candidate_document_ids: list[str] | None = None,
    ) -> dict[str, list[Document]]:
        limit = top_k or self.top_k
        query_tokens = Counter(re.findall(r"[a-z0-9]+", query.lower()))
        path = Path(self.index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Index not found at {path}. Run the indexing stage before inference."
            )
        documents = _read_jsonl_documents(path)
        if candidate_document_ids is not None:
            allowed_ids = set(candidate_document_ids)
            documents = [
                document
                for document in documents
                if _candidate_document_id(document) in allowed_ids
            ]

        scored: list[Document] = []
        for document in documents:
            document_tokens = Counter(re.findall(r"[a-z0-9]+", (document.content or "").lower()))
            score = float(sum((query_tokens & document_tokens).values()))
            scored.append(
                Document(
                    id=document.id,
                    content=document.content,
                    meta=dict(document.meta or {}),
                    score=score,
                )
            )

        scored.sort(key=lambda document: (document.score or 0.0, document.id or ""), reverse=True)
        return {"documents": scored[:limit]}
