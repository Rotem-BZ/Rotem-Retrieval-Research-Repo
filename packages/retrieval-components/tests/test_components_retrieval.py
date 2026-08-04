import pytest
from haystack import Document, Pipeline

import retrieval_components.experimental.elasticsearch_bm25_retriever as retriever_module
import retrieval_components.experimental.elasticsearch_document_indexer as indexer_module
from retrieval_components.dataclasses.query import Query
from retrieval_components.experimental.elasticsearch_bm25_retriever import (
    ElasticsearchBM25Retriever,
)
from retrieval_components.experimental.elasticsearch_document_indexer import (
    ElasticsearchDocumentIndexer,
)
from retrieval_components.ranking.embedding_similarity_ranker import EmbeddingSimilarityRanker


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
                            "body": "retrieval text",
                            "meta": {"source": "mock"},
                        },
                    }
                ]
            }
        }


def test_elasticsearch_components_serialize_content_field_name() -> None:
    pipeline = Pipeline()
    pipeline.add_component(
        "indexer",
        ElasticsearchDocumentIndexer(index_name="docs", content_field_name="body"),
    )
    pipeline.add_component(
        "retriever",
        ElasticsearchBM25Retriever(index_name="docs", content_field_name="body"),
    )

    components = pipeline.to_dict()["components"]
    assert components["indexer"]["init_parameters"]["content_field_name"] == "body"
    assert components["retriever"]["init_parameters"]["content_field_name"] == "body"
    assert "content_field" not in components["indexer"]["init_parameters"]
    assert "content_field" not in components["retriever"]["init_parameters"]


def test_elasticsearch_document_indexer_initializes_client_during_warm_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeElasticsearchClient()
    constructor_calls = []
    monkeypatch.setattr(indexer_module.elasticsearch_import, "check", lambda: None)
    monkeypatch.setattr(
        indexer_module,
        "Elasticsearch",
        lambda hosts: constructor_calls.append(hosts) or client,
        raising=False,
    )
    indexer = ElasticsearchDocumentIndexer(
        index_name="docs",
        hosts=["http://search:9200"],
        content_field_name="body",
        refresh=True,
    )

    with pytest.raises(RuntimeError, match="warmed up"):
        indexer.run([Document(id="d1", content="hello")])

    indexer.warm_up()
    indexer.warm_up()
    assert indexer.run([Document(id="d1", content="hello", meta={"kind": "toy"})]) == {
        "indexed_count": 1
    }
    assert constructor_calls == [["http://search:9200"]]
    assert client.index_calls == [
        {
            "index": "docs",
            "id": "d1",
            "document": {"body": "hello", "meta": {"kind": "toy"}},
            "refresh": True,
        }
    ]


def test_elasticsearch_bm25_retriever_initializes_client_during_warm_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeElasticsearchClient()
    constructor_calls = []
    monkeypatch.setattr(retriever_module.elasticsearch_import, "check", lambda: None)
    monkeypatch.setattr(
        retriever_module,
        "Elasticsearch",
        lambda hosts: constructor_calls.append(hosts) or client,
        raising=False,
    )
    retriever = ElasticsearchBM25Retriever(
        index_name="docs",
        content_field_name="body",
        top_k=3,
    )

    with pytest.raises(RuntimeError, match="warmed up"):
        retriever.run(Query(id="q1", content="retrieval"))

    retriever.warm_up()
    retriever.warm_up()
    result = retriever.run(Query(id="q1", content="retrieval"))

    assert constructor_calls == ["http://localhost:9200"]
    assert client.search_calls == [
        {
            "index": "docs",
            "query": {"match": {"body": "retrieval"}},
            "size": 3,
        }
    ]
    assert result["documents"][0].id == "d1"
    assert result["documents"][0].score == 3.5
    assert result["documents"][0].meta == {"source": "mock"}


def test_elasticsearch_bm25_retriever_filters_to_candidate_ids() -> None:
    client = FakeElasticsearchClient()
    retriever = ElasticsearchBM25Retriever(
        index_name="docs",
        content_field_name="body",
        client=client,
        top_k=3,
    )

    retriever.run(
        Query(id="q1", content="retrieval"),
        candidate_document_ids=["d1", "d2"],
    )

    assert client.search_calls[0] == {
        "index": "docs",
        "query": {
            "bool": {
                "must": [{"match": {"body": "retrieval"}}],
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


def test_embedding_similarity_ranker_scores_embedded_documents() -> None:
    ranker = EmbeddingSimilarityRanker(similarity="cosine")

    result = ranker.run(
        query_embedding=[0.9, 0.1],
        documents=[
            Document(id="near", content="near document", embedding=[1.0, 0.0]),
            Document(id="far", content="far document", embedding=[0.0, 1.0]),
        ],
    )

    assert [document.id for document in result["documents"]] == ["near", "far"]
    assert result["documents"][0].score > 0.9
