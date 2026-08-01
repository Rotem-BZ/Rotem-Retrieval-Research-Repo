"""Persistent Haystack in-memory document indexer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from haystack import Document, component
from haystack.document_stores.in_memory import InMemoryDocumentStore


@component
class PersistedInMemoryDocumentIndexer:
    """Accumulate document batches in memory and atomically persist the completed store."""

    def __init__(
        self,
        output_path: str,
        similarity: Literal["cosine", "dot_product"] = "cosine",
        overwrite: bool = False,
    ) -> None:
        if not output_path.strip():
            raise ValueError("output_path must not be empty.")
        if similarity not in {"cosine", "dot_product"}:
            raise ValueError(f"Unsupported similarity: {similarity}")
        self.output_path = output_path
        self.similarity = similarity
        self.overwrite = overwrite
        self._document_store: InMemoryDocumentStore | None = None

    @asynccontextmanager
    async def write_session(self) -> AsyncIterator[None]:
        """Open an isolated write session and publish it only after successful completion."""

        if self._document_store is not None:
            raise RuntimeError("An index write session is already active.")

        output_path = Path(self.output_path)
        if output_path.exists() and not self.overwrite:
            raise FileExistsError(f"Index already exists and overwrite=false: {output_path}")

        self._document_store = InMemoryDocumentStore(
            embedding_similarity_function=self.similarity,
        )
        try:
            yield
            self._commit()
        finally:
            store = self._document_store
            self._document_store = None
            if store is not None:
                store.delete_all_documents()
                store.shutdown()

    @component.output_types(index_path=str, indexed_count=int)
    def run(self, documents: list[Document]) -> dict[str, str | int]:
        """Write one document batch into the active session."""

        if self._document_store is None:
            raise RuntimeError("PersistedInMemoryDocumentIndexer must run inside write_session().")
        written = self._document_store.write_documents(documents)
        return {
            "index_path": self.output_path,
            "indexed_count": written,
        }

    def _commit(self) -> None:
        store = self._document_store
        if store is None:
            raise RuntimeError("No index write session is active.")

        output_path = Path(self.output_path)
        if output_path.exists() and not self.overwrite:
            raise FileExistsError(f"Index already exists and overwrite=false: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
        try:
            store.save_to_disk(str(temporary_path))
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
