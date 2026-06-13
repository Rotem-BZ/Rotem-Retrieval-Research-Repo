"""Adapter for LangChain text splitters."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from haystack import Document, component


@component
class LangChainDocumentSplitter:
    """Split Haystack documents with a splitter from `langchain_text_splitters`."""

    def __init__(
        self,
        splitter_type: str = "RecursiveCharacterTextSplitter",
        splitter_kwargs: dict[str, Any] | None = None,
        keep_empty: bool = False,
    ) -> None:
        self.splitter_type = splitter_type
        self.splitter_kwargs = splitter_kwargs or {}
        self.keep_empty = keep_empty

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        splitter = self._create_splitter()
        chunks: list[Document] = []

        for document in documents:
            split_texts = self._split_text(splitter, document.content or "")
            if not self.keep_empty:
                split_texts = [text for text in split_texts if text.strip()]

            chunk_count = len(split_texts)
            for chunk_index, text in enumerate(split_texts):
                meta = dict(document.meta or {})
                meta.update(
                    {
                        "source_document_id": document.id,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                    }
                )
                chunks.append(
                    Document(
                        id=_chunk_id(document.id, chunk_index),
                        content=text,
                        meta=meta,
                        score=getattr(document, "score", None),
                    )
                )

        return {"documents": chunks}

    def _create_splitter(self) -> Any:
        try:
            splitters = import_module("langchain_text_splitters")
        except ImportError as exc:
            raise ImportError(
                "LangChainDocumentSplitter requires the optional "
                "`langchain-text-splitters` package."
            ) from exc

        try:
            splitter_cls = getattr(splitters, self.splitter_type)
        except AttributeError as exc:
            raise ValueError(f"Unknown LangChain splitter type: {self.splitter_type}") from exc

        return splitter_cls(**self.splitter_kwargs)

    @staticmethod
    def _split_text(splitter: Any, text: str) -> list[str]:
        if hasattr(splitter, "split_text"):
            return list(splitter.split_text(text))
        if hasattr(splitter, "create_documents"):
            return [document.page_content for document in splitter.create_documents([text])]
        raise TypeError("LangChain splitter must define split_text() or create_documents().")


def _chunk_id(document_id: str | None, chunk_index: int) -> str | None:
    if document_id is None:
        return None
    return f"{document_id}::chunk-{chunk_index}"
