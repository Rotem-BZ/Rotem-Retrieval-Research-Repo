"""JSONL keyword retriever."""

from __future__ import annotations

from pathlib import Path

from haystack import Document, component

from retrieval_research.components.dummy_utils import read_jsonl_documents, tokens


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
        query_tokens = tokens(query)
        documents = _filter_candidates(self._load_documents(), candidate_document_ids)

        scored: list[Document] = []
        for document in documents:
            document_tokens = tokens(document.content or "")
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

        return read_jsonl_documents(path)


def _filter_candidates(
    documents: list[Document],
    candidate_document_ids: list[str] | None,
) -> list[Document]:
    if candidate_document_ids is None:
        return documents

    allowed_ids = set(candidate_document_ids)
    return [
        document
        for document in documents
        if _candidate_document_id(document) in allowed_ids
    ]


def _candidate_document_id(document: Document) -> str | None:
    meta = document.meta or {}
    return meta.get("source_document_id") or document.id
