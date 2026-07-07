"""JSONL-backed embedding retrieval component."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from haystack import Document, component
import numpy as np

from retrieval_research.utils.documents import (
    candidate_document_id,
    copy_document_with_score,
    document_from_record,
)
from retrieval_research.utils.embeddings import (
    embedding_matrix,
    similarity_scores,
    top_score_indices,
)


@component
class JsonlEmbeddingRetriever:
    """Retrieve documents by comparing a query embedding to JSONL document embeddings."""

    def __init__(
        self,
        index_path: str,
        top_k: int = 10,
        similarity: str = "cosine",
    ) -> None:
        self.index_path = index_path
        self.top_k = top_k
        self.similarity = similarity
        self._index: _EmbeddingIndex | None = None

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        candidate_document_ids: list[str] | None = None,
    ) -> dict[str, list[Document]]:
        limit = top_k or self.top_k
        index = self._load_index()
        if not index.documents or limit <= 0:
            return {"documents": []}

        if candidate_document_ids is None:
            candidate_indices = list(range(len(index.documents)))
        else:
            allowed_ids = set(candidate_document_ids)
            candidate_indices = [
                index
                for index, document in enumerate(index.documents)
                if candidate_document_id(document) in allowed_ids
            ]
        if not candidate_indices:
            return {"documents": []}

        scores = similarity_scores(
            query_embedding=query_embedding,
            embeddings=index.embeddings[candidate_indices],
            embedding_norms=index.embedding_norms[candidate_indices],
            similarity=self.similarity,
            dimension_error_context="index embeddings",
        )
        top_indices = top_score_indices(scores, limit)
        return {
            "documents": [
                copy_document_with_score(
                    index.documents[candidate_indices[index_value]],
                    float(scores[index_value]),
                )
                for index_value in top_indices
            ]
        }

    def _load_index(self) -> "_EmbeddingIndex":
        if self._index is not None:
            return self._index

        path = Path(self.index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding index not found at {path}. Run the indexing stage first."
            )

        documents: list[Document] = []
        embeddings: list[list[float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    document = document_from_record(json.loads(line))
                    embedding = getattr(document, "embedding", None)
                    if embedding is not None:
                        documents.append(document)
                        embeddings.append(list(embedding))

        matrix = embedding_matrix(embeddings)
        self._index = _EmbeddingIndex(
            documents=documents,
            embeddings=matrix,
            embedding_norms=np.linalg.norm(matrix, axis=1),
        )
        return self._index


@dataclass(frozen=True)
class _EmbeddingIndex:
    documents: list[Document]
    embeddings: np.ndarray
    embedding_norms: np.ndarray
