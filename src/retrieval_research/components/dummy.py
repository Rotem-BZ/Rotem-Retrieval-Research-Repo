"""Small Haystack components used to exercise the research framework.

These components are deliberately simple. They define the contracts and artifact
shape that richer production components can replace later.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from haystack import Document, component


def _document_from_record(record: dict[str, Any] | Document) -> Document:
    if isinstance(record, Document):
        return record

    return Document(
        id=record.get("id"),
        content=record.get("content", ""),
        meta=dict(record.get("meta") or {}),
    )


def _document_to_record(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "content": document.content,
        "meta": dict(document.meta or {}),
        "score": getattr(document, "score", None),
    }


def _tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


@component
class DummyDocumentSource:
    """Emit configured documents into a Haystack pipeline."""

    def __init__(self, documents: list[dict[str, Any]] | None = None) -> None:
        self.documents = documents or []

    @component.output_types(documents=list[Document])
    def run(self) -> dict[str, list[Document]]:
        return {"documents": [_document_from_record(record) for record in self.documents]}


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

        documents: list[Document] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    documents.append(_document_from_record(json.loads(line)))

        return {"documents": documents}


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
                handle.write(json.dumps(_document_to_record(document), ensure_ascii=False) + "\n")

        return {"index_path": str(path), "indexed_count": len(documents)}


@component
class JsonlKeywordRetriever:
    """A tiny lexical retriever over the JSONL artifact written by the dummy indexer."""

    def __init__(self, index_path: str, top_k: int = 5) -> None:
        self.index_path = index_path
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(self, query: str, top_k: int | None = None) -> dict[str, list[Document]]:
        limit = top_k or self.top_k
        query_tokens = _tokens(query)
        documents = self._load_documents()

        scored: list[Document] = []
        for document in documents:
            document_tokens = _tokens(document.content or "")
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

    def _load_documents(self) -> list[Document]:
        path = Path(self.index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Index not found at {path}. Run the indexing stage before inference."
            )

        documents: list[Document] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    documents.append(_document_from_record(json.loads(line)))
        return documents


@component
class WeightedDocumentFusion:
    """Fuse ranked document lists with weighted reciprocal rank fusion."""

    def __init__(
        self,
        weights: dict[str, float],
        top_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.weights = weights
        self.top_k = top_k
        self.rrf_k = rrf_k

        component.set_input_types(
            self,
            **{source_name: list[Document] for source_name in weights},
        )
        component.set_output_types(self, documents=list[Document])

    def run(self, **ranked_lists: list[Document]) -> dict[str, list[Document]]:
        fused_scores: dict[str, float] = {}
        documents_by_id: dict[str, Document] = {}

        for source_name, weight in self.weights.items():
            documents = ranked_lists.get(source_name, [])
            for rank, document in enumerate(documents, start=1):
                document_id = document.id
                if document_id is None:
                    continue

                documents_by_id.setdefault(document_id, document)
                fused_scores[document_id] = fused_scores.get(document_id, 0.0) + (
                    float(weight) / (self.rrf_k + rank)
                )

        fused = [
            Document(
                id=document.id,
                content=document.content,
                meta=dict(document.meta or {}),
                score=fused_scores[document_id],
            )
            for document_id, document in documents_by_id.items()
        ]
        fused.sort(key=lambda document: (document.score or 0.0, document.id or ""), reverse=True)

        if self.top_k is not None:
            fused = fused[: self.top_k]

        return {"documents": fused}
