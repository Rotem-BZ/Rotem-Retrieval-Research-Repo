"""Haystack components for sparse-autoencoder bi-encoder retrieval."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from haystack import Document, component
from haystack.dataclasses import SparseEmbedding
from torch import Tensor

from sparse_autoencoder_retrieval.model import (
    CompositeCodeSparseAutoencoder,
    load_autoencoder_checkpoint,
)

INDEX_FORMAT = "semantic-sparse-inverted-index-v1"


class _DenseEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_tensor: bool,
        normalize_embeddings: bool,
    ) -> Any: ...


def _load_dense_encoder(model: str, device: str | None) -> _DenseEncoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - declared project dependency
        raise ImportError(
            "Sparse bi-encoder components require sentence-transformers. "
            "Install the sparse-autoencoder-retrieval project environment."
        ) from error
    return SentenceTransformer(model, device=device)


class _SparseBiEncoderBackend:
    """Lazily compose a Sentence Transformer with a trained CCSA projection."""

    def __init__(
        self,
        *,
        model: str,
        autoencoder_path: str,
        device: str | None,
        batch_size: int,
        normalize_embeddings: bool,
        progress_bar: bool,
    ) -> None:
        self.model_name = model
        self.autoencoder_path = autoencoder_path
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.progress_bar = progress_bar
        self._dense_encoder: _DenseEncoder | None = None
        self._autoencoder: CompositeCodeSparseAutoencoder | None = None

    def warm_up(self) -> None:
        if self._dense_encoder is None:
            self._dense_encoder = _load_dense_encoder(self.model_name, self.device)
        if self._autoencoder is None:
            self._autoencoder = load_autoencoder_checkpoint(
                self.autoencoder_path,
                device=self.device or "cpu",
            )

    def encode(self, texts: list[str], *, role: Literal["query", "document"]) -> list[SparseEmbedding]:
        if not texts:
            return []
        self.warm_up()
        assert self._dense_encoder is not None
        assert self._autoencoder is not None

        method = getattr(self._dense_encoder, f"encode_{role}", None)
        if method is None:
            method = self._dense_encoder.encode
        dense_vectors = method(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        tensor = torch.as_tensor(dense_vectors, dtype=torch.float32, device=self.device or "cpu")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] != self._autoencoder.input_dim:
            raise ValueError(
                "Bi-encoder output dimension does not match the sparse autoencoder: "
                f"model produced {tensor.shape[1]}, checkpoint expects "
                f"{self._autoencoder.input_dim}."
            )
        with torch.inference_mode():
            sparse_codes = self._autoencoder.encode(tensor, stochastic=False).cpu()
        return [_to_sparse_embedding(code) for code in sparse_codes]


def _to_sparse_embedding(code: Tensor) -> SparseEmbedding:
    indices = torch.nonzero(code, as_tuple=False).flatten().tolist()
    values = code[indices].tolist()
    return SparseEmbedding(indices=indices, values=values)


def _validate_embedder_settings(*, model: str, autoencoder_path: str, batch_size: int) -> None:
    if not model.strip():
        raise ValueError("model must not be empty.")
    if not autoencoder_path.strip():
        raise ValueError("autoencoder_path must not be empty.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")


@component
class SparseAutoencoderTextEmbedder:
    """Encode query text with a bi-encoder followed by a trained CCSA."""

    def __init__(
        self,
        model: str,
        autoencoder_path: str,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        progress_bar: bool = False,
    ) -> None:
        _validate_embedder_settings(
            model=model,
            autoencoder_path=autoencoder_path,
            batch_size=batch_size,
        )
        self.model = model
        self.autoencoder_path = autoencoder_path
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.progress_bar = progress_bar
        self._backend = _SparseBiEncoderBackend(
            model=model,
            autoencoder_path=autoencoder_path,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            progress_bar=progress_bar,
        )

    def warm_up(self) -> None:
        """Load the dense encoder and CCSA checkpoint once."""

        self._backend.warm_up()

    @component.output_types(sparse_embedding=SparseEmbedding)
    def run(self, text: str) -> dict[str, SparseEmbedding]:
        if not text.strip():
            raise ValueError("text must not be empty.")
        sparse_embedding = self._backend.encode([text], role="query")[0]
        return {"sparse_embedding": sparse_embedding}


@component
class SparseAutoencoderDocumentEmbedder:
    """Attach CCSA semantic sparse vectors to copies of Haystack documents."""

    def __init__(
        self,
        model: str,
        autoencoder_path: str,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        progress_bar: bool = False,
    ) -> None:
        _validate_embedder_settings(
            model=model,
            autoencoder_path=autoencoder_path,
            batch_size=batch_size,
        )
        self.model = model
        self.autoencoder_path = autoencoder_path
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.progress_bar = progress_bar
        self._backend = _SparseBiEncoderBackend(
            model=model,
            autoencoder_path=autoencoder_path,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            progress_bar=progress_bar,
        )

    def warm_up(self) -> None:
        """Load the dense encoder and CCSA checkpoint once."""

        self._backend.warm_up()

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        texts: list[str] = []
        for document in documents:
            if document.content is None or not document.content.strip():
                raise ValueError(f"Document {document.id!r} has no text content to encode.")
            texts.append(document.content)
        sparse_embeddings = self._backend.encode(texts, role="document")
        return {
            "documents": [
                replace(document, sparse_embedding=sparse_embedding)
                for document, sparse_embedding in zip(
                    documents,
                    sparse_embeddings,
                    strict=True,
                )
            ]
        }


def _validated_sparse_pairs(
    sparse_embedding: SparseEmbedding | None,
    *,
    owner: str,
) -> list[tuple[int, float]]:
    if sparse_embedding is None:
        raise ValueError(f"{owner} has no sparse_embedding.")
    if len(sparse_embedding.indices) != len(sparse_embedding.values):
        raise ValueError(f"{owner} has mismatched sparse indices and values.")

    pairs: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_index, raw_value in zip(
        sparse_embedding.indices,
        sparse_embedding.values,
        strict=True,
    ):
        index = int(raw_index)
        value = float(raw_value)
        if index < 0:
            raise ValueError(f"{owner} contains a negative sparse dimension: {index}.")
        if index in seen:
            raise ValueError(f"{owner} contains duplicate sparse dimension {index}.")
        if not math.isfinite(value) or value == 0.0:
            raise ValueError(f"{owner} contains a non-finite or zero sparse weight.")
        seen.add(index)
        pairs.append((index, value))
    return pairs


def _document_record(document: Document, *, store_dense_embeddings: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": document.id,
        "content": document.content,
        "meta": dict(document.meta or {}),
        "score": document.score,
    }
    if store_dense_embeddings and document.embedding is not None:
        embedding = document.embedding
        if hasattr(embedding, "tolist") and callable(embedding.tolist):
            embedding = embedding.tolist()
        record["embedding"] = embedding
    return record


def _write_semantic_sparse_index(
    path: Path,
    *,
    records: list[dict[str, Any]],
    postings: dict[int, list[list[int | float]]],
    sparse_dimension: int | None,
) -> None:
    payload = {
        "format": INDEX_FORMAT,
        "documents": records,
        "postings": {
            str(dimension): values for dimension, values in sorted(postings.items())
        },
        "statistics": {
            "document_count": len(records),
            "sparse_dimension": sparse_dimension
            if sparse_dimension is not None
            else (max(postings, default=-1) + 1),
            "active_dimension_count": len(postings),
            "posting_count": sum(len(values) for values in postings.values()),
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


@component
class SemanticSparseIndexer:
    """Create a lexical-style postings index from semantic sparse vectors."""

    def __init__(
        self,
        output_path: str,
        overwrite: bool = False,
        store_dense_embeddings: bool = False,
        sparse_dimension: int | None = None,
    ) -> None:
        if not output_path.strip():
            raise ValueError("output_path must not be empty.")
        if sparse_dimension is not None and sparse_dimension <= 0:
            raise ValueError("sparse_dimension must be greater than zero when provided.")
        self.output_path = output_path
        self.overwrite = overwrite
        self.store_dense_embeddings = store_dense_embeddings
        self.sparse_dimension = sparse_dimension
        self._postings: dict[int, list[list[int | float]]] | None = None
        self._records: list[dict[str, Any]] | None = None
        self._seen_ids: set[str] | None = None

    @component.output_types(index_path=str, indexed_count=int)
    def run(
        self,
        documents: list[Document],
        append: bool = False,
    ) -> dict[str, str | int]:
        path = Path(self.output_path)
        if not append and path.exists() and not self.overwrite:
            raise FileExistsError(f"Index already exists and overwrite=false: {path}")

        if not append:
            self._postings = {}
            self._records = []
            self._seen_ids = set()
        if self._postings is None or self._records is None or self._seen_ids is None:
            raise RuntimeError("append=True requires an earlier non-appending write.")

        postings = self._postings
        records = self._records
        seen_ids = self._seen_ids
        first_ordinal = len(records)
        for ordinal, document in enumerate(documents, start=first_ordinal):
            if document.id in seen_ids:
                raise ValueError(f"Duplicate document id in sparse index: {document.id!r}.")
            seen_ids.add(document.id)
            for dimension, weight in _validated_sparse_pairs(
                document.sparse_embedding,
                owner=f"Document {document.id!r}",
            ):
                if self.sparse_dimension is not None and dimension >= self.sparse_dimension:
                    raise ValueError(
                        f"Document {document.id!r} uses sparse dimension {dimension}, "
                        f"outside configured size {self.sparse_dimension}."
                    )
                postings.setdefault(dimension, []).append([ordinal, weight])
            records.append(
                _document_record(
                    document,
                    store_dense_embeddings=self.store_dense_embeddings,
                )
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        _write_semantic_sparse_index(
            path,
            records=records,
            postings=postings,
            sparse_dimension=self.sparse_dimension,
        )
        return {"index_path": str(path), "indexed_count": len(documents)}


def _candidate_document_id(document: Document) -> str:
    return str((document.meta or {}).get("source_document_id") or document.id)


@component
class SemanticSparseRetriever:
    """Retrieve documents by accumulating sparse dot products over posting lists."""

    def __init__(
        self,
        index_path: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> None:
        if not index_path.strip():
            raise ValueError("index_path must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not math.isfinite(min_score):
            raise ValueError("min_score must be finite.")
        self.index_path = index_path
        self.top_k = top_k
        self.min_score = min_score
        self._index: _SemanticSparseIndex | None = None

    @component.output_types(documents=list[Document])
    def run(
        self,
        sparse_embedding: SparseEmbedding,
        top_k: int | None = None,
        candidate_document_ids: list[str] | None = None,
    ) -> dict[str, list[Document]]:
        limit = self.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than zero.")
        query_pairs = _validated_sparse_pairs(sparse_embedding, owner="Query")
        index = self._load_index()

        allowed_ordinals: set[int] | None = None
        if candidate_document_ids:
            allowed_ids = set(candidate_document_ids)
            allowed_ordinals = {
                ordinal
                for ordinal, document in enumerate(index.documents)
                if _candidate_document_id(document) in allowed_ids
            }
            if not allowed_ordinals:
                return {"documents": []}

        scores: dict[int, float] = {}
        for dimension, query_weight in query_pairs:
            for ordinal, document_weight in index.postings.get(dimension, ()):
                if allowed_ordinals is None or ordinal in allowed_ordinals:
                    scores[ordinal] = scores.get(ordinal, 0.0) + query_weight * document_weight

        ranked = sorted(
            (
                (ordinal, score)
                for ordinal, score in scores.items()
                if score > self.min_score
            ),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
        return {
            "documents": [
                replace(index.documents[ordinal], score=float(score))
                for ordinal, score in ranked
            ]
        }

    def _load_index(self) -> _SemanticSparseIndex:
        if self._index is not None:
            return self._index
        path = Path(self.index_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Semantic sparse index not found at {path}. Run the indexing stage first."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != INDEX_FORMAT:
            raise ValueError(f"Unsupported semantic sparse index format: {path}")

        records = payload.get("documents")
        raw_postings = payload.get("postings")
        if not isinstance(records, list) or not isinstance(raw_postings, dict):
            raise TypeError(f"Malformed semantic sparse index: {path}")
        postings: dict[int, tuple[tuple[int, float], ...]] = {}
        sparse_by_document: list[list[tuple[int, float]]] = [[] for _ in records]
        for raw_dimension, raw_values in raw_postings.items():
            dimension = int(raw_dimension)
            parsed: list[tuple[int, float]] = []
            for raw_ordinal, raw_weight in raw_values:
                ordinal = int(raw_ordinal)
                weight = float(raw_weight)
                if ordinal < 0 or ordinal >= len(records):
                    raise ValueError(f"Posting references invalid document ordinal {ordinal}.")
                parsed.append((ordinal, weight))
                sparse_by_document[ordinal].append((dimension, weight))
            postings[dimension] = tuple(parsed)

        documents = [
            Document(
                id=record["id"],
                content=record.get("content"),
                meta=dict(record.get("meta") or {}),
                score=record.get("score"),
                embedding=record.get("embedding"),
                sparse_embedding=SparseEmbedding(
                    indices=[dimension for dimension, _ in sparse_by_document[ordinal]],
                    values=[weight for _, weight in sparse_by_document[ordinal]],
                ),
            )
            for ordinal, record in enumerate(records)
        ]
        self._index = _SemanticSparseIndex(documents=documents, postings=postings)
        return self._index


class _SemanticSparseIndex:
    def __init__(
        self,
        *,
        documents: list[Document],
        postings: dict[int, tuple[tuple[int, float], ...]],
    ) -> None:
        self.documents = documents
        self.postings = postings
