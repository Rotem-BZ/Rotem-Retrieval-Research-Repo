import json
from pathlib import Path
from typing import Any

import pytest
import torch
from haystack import Document
from haystack.dataclasses import SparseEmbedding

import sparse_autoencoder_retrieval.components as component_module
from sparse_autoencoder_retrieval import (
    CompositeCodeSparseAutoencoder,
    SemanticSparseIndexer,
    SemanticSparseRetriever,
    SparseAutoencoderDocumentEmbedder,
    SparseAutoencoderTextEmbedder,
)
from sparse_autoencoder_retrieval.model import save_autoencoder_checkpoint


class _FakeBiEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def encode_query(self, texts: list[str], **_: Any) -> torch.Tensor:
        self.calls.append(("query", texts))
        return torch.tensor([[1.0, 0.0, -1.0] for _ in texts])

    def encode_document(self, texts: list[str], **_: Any) -> torch.Tensor:
        self.calls.append(("document", texts))
        return torch.tensor([[0.0, 1.0, -1.0] for _ in texts])


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    model = CompositeCodeSparseAutoencoder(
        input_dim=3,
        num_codebooks=2,
        codebook_size=3,
        use_batch_norm=False,
    )
    return save_autoencoder_checkpoint(model, tmp_path / "ccsa.pt")


def test_text_embedder_uses_query_path_and_returns_native_sparse_embedding(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
) -> None:
    dense_encoder = _FakeBiEncoder()
    monkeypatch.setattr(
        component_module,
        "_load_dense_encoder",
        lambda model, device: dense_encoder,
    )
    embedder = SparseAutoencoderTextEmbedder(
        model="fake-bi-encoder",
        autoencoder_path=str(checkpoint_path),
    )

    result = embedder.run("semantic query")["sparse_embedding"]

    assert dense_encoder.calls == [("query", ["semantic query"])]
    assert isinstance(result, SparseEmbedding)
    assert len(result.indices) == 2
    assert result.values == [1.0, 1.0]


def test_document_embedder_preserves_fields_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
) -> None:
    dense_encoder = _FakeBiEncoder()
    monkeypatch.setattr(
        component_module,
        "_load_dense_encoder",
        lambda model, device: dense_encoder,
    )
    original = Document(
        id="doc-1",
        content="semantic document",
        meta={"source": "fixture"},
        score=0.75,
        embedding=[0.1, 0.2, 0.3],
    )
    embedder = SparseAutoencoderDocumentEmbedder(
        model="fake-bi-encoder",
        autoencoder_path=str(checkpoint_path),
    )

    result = embedder.run([original])["documents"][0]

    assert dense_encoder.calls == [("document", ["semantic document"])]
    assert result is not original
    assert result.id == original.id
    assert result.content == original.content
    assert result.meta == original.meta
    assert result.score == original.score
    assert result.embedding == original.embedding
    assert result.sparse_embedding is not None
    assert len(result.sparse_embedding.indices) == 2
    assert original.sparse_embedding is None


def test_embedder_rejects_checkpoint_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
) -> None:
    class WrongDimensionEncoder(_FakeBiEncoder):
        def encode_query(self, texts: list[str], **_: Any) -> torch.Tensor:
            return torch.ones((len(texts), 2))

    monkeypatch.setattr(
        component_module,
        "_load_dense_encoder",
        lambda model, device: WrongDimensionEncoder(),
    )
    embedder = SparseAutoencoderTextEmbedder(
        model="fake-bi-encoder",
        autoencoder_path=str(checkpoint_path),
    )

    with pytest.raises(ValueError, match="output dimension"):
        embedder.run("query")


def _document(
    document_id: str,
    *,
    indices: list[int],
    values: list[float],
    source_id: str | None = None,
) -> Document:
    meta = {"source_document_id": source_id} if source_id is not None else {}
    return Document(
        id=document_id,
        content=f"content {document_id}",
        meta=meta,
        sparse_embedding=SparseEmbedding(indices=indices, values=values),
    )


def test_semantic_sparse_index_and_retrieval_round_trip(tmp_path: Path) -> None:
    index_path = tmp_path / "semantic-sparse.json"
    documents = [
        _document("doc-1", indices=[0, 2], values=[1.0, 1.0]),
        _document("doc-2", indices=[1, 2], values=[1.0, 1.0]),
        _document("chunk-3", indices=[0, 3], values=[1.0, 1.0], source_id="doc-3"),
    ]

    index_result = SemanticSparseIndexer(output_path=str(index_path)).run(documents)
    retrieval_result = SemanticSparseRetriever(
        index_path=str(index_path),
        top_k=3,
    ).run(SparseEmbedding(indices=[0, 2], values=[1.0, 0.5]))

    assert index_result == {"index_path": str(index_path), "indexed_count": 3}
    assert [document.id for document in retrieval_result["documents"]] == [
        "doc-1",
        "chunk-3",
        "doc-2",
    ]
    assert [document.score for document in retrieval_result["documents"]] == [1.5, 1.0, 0.5]
    assert all(document.sparse_embedding is not None for document in retrieval_result["documents"])

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["format"] == "semantic-sparse-inverted-index-v1"
    assert payload["postings"]["0"] == [[0, 1.0], [2, 1.0]]
    assert payload["statistics"] == {
        "document_count": 3,
        "sparse_dimension": 4,
        "active_dimension_count": 4,
        "posting_count": 6,
    }


def test_semantic_sparse_retriever_filters_by_source_document_id(tmp_path: Path) -> None:
    index_path = tmp_path / "semantic-sparse.json"
    documents = [
        _document("doc-1", indices=[0], values=[1.0]),
        _document("chunk-2", indices=[0], values=[1.0], source_id="source-2"),
    ]
    SemanticSparseIndexer(output_path=str(index_path)).run(documents)
    retriever = SemanticSparseRetriever(index_path=str(index_path))

    result = retriever.run(
        SparseEmbedding(indices=[0], values=[1.0]),
        candidate_document_ids=["source-2"],
    )

    assert [document.id for document in result["documents"]] == ["chunk-2"]


def test_empty_candidate_list_means_unrestricted_retrieval(tmp_path: Path) -> None:
    index_path = tmp_path / "semantic-sparse.json"
    documents = [
        _document("doc-1", indices=[0], values=[1.0]),
        _document("doc-2", indices=[0], values=[1.0]),
    ]
    SemanticSparseIndexer(output_path=str(index_path)).run(documents)

    result = SemanticSparseRetriever(index_path=str(index_path)).run(
        SparseEmbedding(indices=[0], values=[1.0]),
        candidate_document_ids=[],
    )

    assert [document.id for document in result["documents"]] == ["doc-1", "doc-2"]


def test_semantic_sparse_indexer_requires_sparse_vectors(tmp_path: Path) -> None:
    indexer = SemanticSparseIndexer(output_path=str(tmp_path / "index.json"))

    with pytest.raises(ValueError, match="has no sparse_embedding"):
        indexer.run([Document(id="doc-1", content="missing vector")])


def test_semantic_sparse_indexer_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    indexer = SemanticSparseIndexer(output_path=str(tmp_path / "index.json"))
    documents = [
        _document("same", indices=[0], values=[1.0]),
        _document("same", indices=[1], values=[1.0]),
    ]

    with pytest.raises(ValueError, match="Duplicate document id"):
        indexer.run(documents)


def test_semantic_sparse_indexer_validates_configured_dimension(tmp_path: Path) -> None:
    indexer = SemanticSparseIndexer(
        output_path=str(tmp_path / "index.json"),
        sparse_dimension=2,
    )

    with pytest.raises(ValueError, match="outside configured size"):
        indexer.run([_document("doc-1", indices=[2], values=[1.0])])


def test_semantic_sparse_indexer_preserves_ordinals_across_batches(
    tmp_path: Path,
) -> None:
    configured_path = tmp_path / "configured.json"
    temporary_path = tmp_path / ".configured.json.batch.tmp"
    indexer = SemanticSparseIndexer(output_path=str(configured_path))

    indexer.begin_batch_write(str(temporary_path))
    indexer.run([_document("doc-1", indices=[0], values=[1.0])])
    indexer.run([_document("doc-2", indices=[0, 1], values=[0.5, 1.0])])
    indexer.finish_batch_write()

    payload = json.loads(temporary_path.read_text(encoding="utf-8"))
    assert [record["id"] for record in payload["documents"]] == ["doc-1", "doc-2"]
    assert payload["postings"]["0"] == [[0, 1.0], [1, 0.5]]
    assert payload["postings"]["1"] == [[1, 1.0]]
    assert payload["statistics"]["document_count"] == 2
    assert not configured_path.exists()


def test_semantic_sparse_indexer_rejects_duplicates_across_batches(
    tmp_path: Path,
) -> None:
    indexer = SemanticSparseIndexer(output_path=str(tmp_path / "configured.json"))
    temporary_path = tmp_path / ".configured.json.batch.tmp"
    indexer.begin_batch_write(str(temporary_path))
    indexer.run([_document("same", indices=[0], values=[1.0])])

    with pytest.raises(ValueError, match="Duplicate document id"):
        indexer.run([_document("same", indices=[1], values=[1.0])])

    indexer.abort_batch_write()
