"""Embedding retrieval from a persisted Haystack in-memory document store."""

from __future__ import annotations

import asyncio
from threading import Lock

from haystack import Document, component
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


def _candidate_filters(candidate_document_ids: list[str]) -> dict:
    return {
        "operator": "OR",
        "conditions": [
            {
                "field": "id",
                "operator": "in",
                "value": candidate_document_ids,
            },
            {
                "field": "meta.source_document_id",
                "operator": "in",
                "value": candidate_document_ids,
            },
        ],
    }


@component
class PersistedInMemoryEmbeddingRetriever:
    """Load a persisted store once and delegate search to Haystack's native retriever."""

    def __init__(
        self,
        index_path: str,
        top_k: int = 10,
        return_embedding: bool = True,
    ) -> None:
        if not index_path.strip():
            raise ValueError("index_path must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.index_path = index_path
        self.top_k = top_k
        self.return_embedding = return_embedding
        self._retriever: InMemoryEmbeddingRetriever | None = None
        self._warm_up_lock = Lock()

    def warm_up(self) -> None:
        """Load the persisted document store once."""

        if self._retriever is not None:
            return
        with self._warm_up_lock:
            if self._retriever is not None:
                return
            store = InMemoryDocumentStore.load_from_disk(self.index_path)
            self._retriever = InMemoryEmbeddingRetriever(
                document_store=store,
                top_k=self.top_k,
                return_embedding=self.return_embedding,
            )

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        candidate_document_ids: list[str] | None = None,
    ) -> dict[str, list[Document]]:
        """Retrieve documents, optionally restricted to candidate source IDs."""

        if candidate_document_ids is not None and not candidate_document_ids:
            return {"documents": []}
        self.warm_up()
        if self._retriever is None:  # pragma: no cover - guarded by warm_up
            raise RuntimeError("Retriever warm-up did not initialize the document store.")
        return self._retriever.run(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=(
                _candidate_filters(candidate_document_ids)
                if candidate_document_ids is not None
                else None
            ),
        )

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        candidate_document_ids: list[str] | None = None,
    ) -> dict[str, list[Document]]:
        """Asynchronously retrieve documents through Haystack's native retriever."""

        if candidate_document_ids is not None and not candidate_document_ids:
            return {"documents": []}
        await asyncio.to_thread(self.warm_up)
        if self._retriever is None:  # pragma: no cover - guarded by warm_up
            raise RuntimeError("Retriever warm-up did not initialize the document store.")
        return await self._retriever.run_async(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=(
                _candidate_filters(candidate_document_ids)
                if candidate_document_ids is not None
                else None
            ),
        )
