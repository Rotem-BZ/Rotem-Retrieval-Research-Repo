import json
from pathlib import Path

from haystack import Document

from retrieval_research.components.dummy import JsonlDocumentIndexer, JsonlKeywordRetriever
from retrieval_research.components.retrieval import (
    ElasticsearchBM25Retriever,
    ElasticsearchDocumentIndexer,
    JsonlEmbeddingRetriever,
)


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.index_calls = []
        self.search_calls = []

    def index(self, **kwargs):
        self.index_calls.append(kwargs)
        return {"result": "created"}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "d1",
                        "_score": 3.5,
                        "_source": {
                            "content": "retrieval text",
                            "meta": {"source": "mock"},
                        },
                    }
                ]
            }
        }


def test_elasticsearch_document_indexer_uses_injected_client() -> None:
    client = FakeElasticsearchClient()
    indexer = ElasticsearchDocumentIndexer(index_name="docs", client=client, refresh=True)

    assert indexer.run([Document(id="d1", content="hello", meta={"kind": "toy"})]) == {
        "indexed_count": 1
    }
    assert client.index_calls == [
        {
            "index": "docs",
            "id": "d1",
            "document": {"content": "hello", "meta": {"kind": "toy"}},
            "refresh": True,
        }
    ]


def test_elasticsearch_bm25_retriever_uses_injected_client() -> None:
    client = FakeElasticsearchClient()
    retriever = ElasticsearchBM25Retriever(index_name="docs", client=client, top_k=3)

    result = retriever.run("retrieval")

    assert client.search_calls == [
        {
            "index": "docs",
            "query": {"match": {"content": "retrieval"}},
            "size": 3,
        }
    ]
    assert result["documents"][0].id == "d1"
    assert result["documents"][0].score == 3.5
    assert result["documents"][0].meta == {"source": "mock"}


def test_elasticsearch_bm25_retriever_filters_to_candidate_ids() -> None:
    client = FakeElasticsearchClient()
    retriever = ElasticsearchBM25Retriever(index_name="docs", client=client, top_k=3)

    retriever.run("retrieval", candidate_document_ids=["d1", "d2"])

    assert client.search_calls[0] == {
        "index": "docs",
        "query": {
            "bool": {
                "must": [{"match": {"content": "retrieval"}}],
                "filter": [
                    {
                        "bool": {
                            "should": [
                                {"ids": {"values": ["d1", "d2"]}},
                                {"terms": {"meta.source_document_id": ["d1", "d2"]}},
                                {"terms": {"meta.source_document_id.keyword": ["d1", "d2"]}},
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                ],
            }
        },
        "size": 3,
    }


def test_jsonl_keyword_retriever_filters_to_candidate_ids(tmp_path: Path) -> None:
    index_path = tmp_path / "documents.jsonl"
    indexer = JsonlDocumentIndexer(output_path=str(index_path))
    indexer.run(
        [
            Document(id="near", content="retrieval retrieval"),
            Document(id="far", content="retrieval"),
        ]
    )

    retriever = JsonlKeywordRetriever(index_path=str(index_path), top_k=2)
    result = retriever.run("retrieval", candidate_document_ids=["far"])

    assert [document.id for document in result["documents"]] == ["far"]


def test_jsonl_embedding_retriever_reads_persisted_embeddings(tmp_path: Path) -> None:
    index_path = tmp_path / "embeddings.jsonl"
    indexer = JsonlDocumentIndexer(output_path=str(index_path))
    indexer.run(
        [
            Document(id="near", content="near document", embedding=[1.0, 0.0]),
            Document(id="far", content="far document", embedding=[0.0, 1.0]),
        ]
    )

    records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["embedding"] == [1.0, 0.0]

    retriever = JsonlEmbeddingRetriever(index_path=str(index_path), top_k=1)
    result = retriever.run(query_embedding=[0.9, 0.1])

    assert [document.id for document in result["documents"]] == ["near"]


def test_jsonl_embedding_retriever_filters_to_candidate_ids(tmp_path: Path) -> None:
    index_path = tmp_path / "embeddings.jsonl"
    indexer = JsonlDocumentIndexer(output_path=str(index_path))
    indexer.run(
        [
            Document(id="near", content="near document", embedding=[1.0, 0.0]),
            Document(id="far", content="far document", embedding=[0.0, 1.0]),
        ]
    )

    retriever = JsonlEmbeddingRetriever(index_path=str(index_path), top_k=2)
    result = retriever.run(query_embedding=[0.9, 0.1], candidate_document_ids=["far"])

    assert [document.id for document in result["documents"]] == ["far"]


def test_jsonl_embedding_retriever_filters_chunks_by_source_document_id(tmp_path: Path) -> None:
    index_path = tmp_path / "embeddings.jsonl"
    indexer = JsonlDocumentIndexer(output_path=str(index_path))
    indexer.run(
        [
            Document(
                id="d1::chunk-0",
                content="near chunk",
                meta={"source_document_id": "d1"},
                embedding=[1.0, 0.0],
            ),
            Document(
                id="d2::chunk-0",
                content="far chunk",
                meta={"source_document_id": "d2"},
                embedding=[0.0, 1.0],
            ),
        ]
    )

    retriever = JsonlEmbeddingRetriever(index_path=str(index_path), top_k=2)
    result = retriever.run(query_embedding=[0.9, 0.1], candidate_document_ids=["d2"])

    assert [document.id for document in result["documents"]] == ["d2::chunk-0"]
