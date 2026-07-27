"""Adapter for LangChain text splitters."""

from __future__ import annotations

from haystack import Document, component
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


@component
class LangChainDocumentSplitter:
    """Recursively split Haystack documents by character or tokenizer length."""

    def __init__(
        self,
        chunk_size: int = 360,
        chunk_overlap: int = 60,
        tokenizer_path: str | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer_path = tokenizer_path
        self._splitter = None

    def warm_up(self) -> None:
        """Load the recursive splitter and optional tokenizer once."""
        if self._splitter is not None:
            return

        splitter_kwargs = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if self.tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            splitter_kwargs["length_function"] = lambda text: len(
                tokenizer.encode(text, add_special_tokens=False)
            )

        self._splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        self.warm_up()
        chunks: list[Document] = []

        for document in documents:
            if document.content is None:
                raise ValueError(
                    f"LangChainDocumentSplitter requires document {document.id!r} to have content."
                )
            if document.id is None:
                raise ValueError("LangChainDocumentSplitter requires every document to have an id.")
            split_texts = [
                text for text in self._splitter.split_text(document.content) if text.strip()
            ]

            chunk_count = len(split_texts)
            for chunk_index, text in enumerate(split_texts):
                meta = dict(document.meta)
                meta.update(
                    {
                        "source_document_id": document.id,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                    }
                )
                chunks.append(
                    Document(
                        id=f"{document.id}::chunk-{chunk_index}",
                        content=text,
                        meta=meta,
                        score=document.score,
                    )
                )

        return {"documents": chunks}
