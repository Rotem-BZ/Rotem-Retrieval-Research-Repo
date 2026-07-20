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
        splitter = splitter_cls(**self.splitter_kwargs)
        chunks: list[Document] = []

        for document in documents:
            text = document.content or ""
            if hasattr(splitter, "split_text"):
                split_texts = list(splitter.split_text(text))
            elif hasattr(splitter, "create_documents"):
                split_texts = [item.page_content for item in splitter.create_documents([text])]
            else:
                raise TypeError(
                    "LangChain splitter must define split_text() or create_documents()."
                )
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
                        id=(None if document.id is None else f"{document.id}::chunk-{chunk_index}"),
                        content=text,
                        meta=meta,
                        score=getattr(document, "score", None),
                    )
                )

        return {"documents": chunks}
