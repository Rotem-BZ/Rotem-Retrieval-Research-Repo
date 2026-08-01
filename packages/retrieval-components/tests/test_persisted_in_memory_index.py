import asyncio
from pathlib import Path

import pytest
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore

from retrieval_components.indexing import PersistedInMemoryDocumentIndexer
from retrieval_components.retrieval import PersistedInMemoryEmbeddingRetriever


def test_persisted_indexer_commits_batches(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    indexer = PersistedInMemoryDocumentIndexer(
        output_path=str(index_path),
        similarity="cosine",
    )

    async def write_index() -> None:
        async with indexer.write_session():
            first = indexer.run([Document(id="d1", content="one", embedding=[1.0, 0.0])])
            second = indexer.run([Document(id="d2", content="two", embedding=[0.0, 1.0])])
            assert first["indexed_count"] == 1
            assert second["indexed_count"] == 1

    asyncio.run(write_index())

    store = InMemoryDocumentStore.load_from_disk(str(index_path))
    try:
        assert store.embedding_similarity_function == "cosine"
        assert [document.id for document in store.filter_documents()] == ["d1", "d2"]
    finally:
        store.delete_all_documents()
        store.shutdown()


def test_persisted_indexer_requires_a_write_session(tmp_path: Path) -> None:
    indexer = PersistedInMemoryDocumentIndexer(str(tmp_path / "index.json"))

    with pytest.raises(RuntimeError, match="inside write_session"):
        indexer.run([Document(id="d1", content="one")])


def test_persisted_indexer_rolls_back_failed_session(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    indexer = PersistedInMemoryDocumentIndexer(str(index_path))

    async def fail_write() -> None:
        with pytest.raises(RuntimeError, match="batch failed"):
            async with indexer.write_session():
                indexer.run([Document(id="d1", content="one")])
                raise RuntimeError("batch failed")

    asyncio.run(fail_write())

    assert not index_path.exists()
    assert not list(tmp_path.glob(".index.json.*.tmp"))


def test_persisted_indexer_rolls_back_cancelled_session(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    indexer = PersistedInMemoryDocumentIndexer(str(index_path))

    async def cancel_write() -> None:
        entered = asyncio.Event()

        async def write() -> None:
            async with indexer.write_session():
                indexer.run([Document(id="d1", content="one")])
                entered.set()
                await asyncio.Future()

        task = asyncio.create_task(write())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_write())

    assert not index_path.exists()
    assert not list(tmp_path.glob(".index.json.*.tmp"))


def test_persisted_retriever_delegates_cosine_search_and_candidate_filters(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    store.write_documents(
        [
            Document(id="near", content="near", embedding=[1.0, 0.0]),
            Document(
                id="far::chunk-0",
                content="far",
                meta={"source_document_id": "far"},
                embedding=[0.0, 1.0],
            ),
        ]
    )
    store.save_to_disk(str(index_path))
    retriever = PersistedInMemoryEmbeddingRetriever(str(index_path), top_k=2)

    all_documents = retriever.run(query_embedding=[0.9, 0.1])["documents"]
    direct_candidate = retriever.run(query_embedding=[0.9, 0.1], candidate_document_ids=["near"])[
        "documents"
    ]
    source_candidate = retriever.run(query_embedding=[0.9, 0.1], candidate_document_ids=["far"])[
        "documents"
    ]

    assert [document.id for document in all_documents] == ["near", "far::chunk-0"]
    assert [document.id for document in direct_candidate] == ["near"]
    assert [document.id for document in source_candidate] == ["far::chunk-0"]


def test_persisted_retriever_returns_no_documents_for_empty_candidates(
    tmp_path: Path,
) -> None:
    retriever = PersistedInMemoryEmbeddingRetriever(str(tmp_path / "missing.json"))

    assert retriever.run(query_embedding=[1.0, 0.0], candidate_document_ids=[]) == {"documents": []}


@pytest.mark.parametrize(
    ("component_type", "argument"),
    [
        (PersistedInMemoryDocumentIndexer, "output_path"),
        (PersistedInMemoryEmbeddingRetriever, "index_path"),
    ],
)
def test_persisted_components_reject_empty_paths(component_type, argument: str) -> None:
    with pytest.raises(ValueError, match=f"{argument} must not be empty"):
        component_type(**{argument: " "})


def test_persisted_components_serialize_only_constructor_configuration(
    tmp_path: Path,
) -> None:
    pipeline = Pipeline()
    pipeline.add_component(
        "indexer",
        PersistedInMemoryDocumentIndexer(
            output_path=str(tmp_path / "index.json"),
            similarity="dot_product",
            overwrite=True,
        ),
    )
    pipeline.add_component(
        "retriever",
        PersistedInMemoryEmbeddingRetriever(
            index_path=str(tmp_path / "index.json"),
            top_k=7,
            return_embedding=False,
        ),
    )

    components = pipeline.to_dict()["components"]
    assert components["indexer"]["init_parameters"] == {
        "output_path": str(tmp_path / "index.json"),
        "similarity": "dot_product",
        "overwrite": True,
    }
    assert components["retriever"]["init_parameters"] == {
        "index_path": str(tmp_path / "index.json"),
        "top_k": 7,
        "return_embedding": False,
    }
