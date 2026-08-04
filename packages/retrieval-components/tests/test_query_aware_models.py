from __future__ import annotations

from typing import Any

import pytest
from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder as HaystackSentenceTransformersTextEmbedder,
)
from haystack.components.rankers import (
    SentenceTransformersSimilarityRanker as HaystackSentenceTransformersSimilarityRanker,
)
from haystack.components.rankers import (
    TransformersSimilarityRanker as HaystackTransformersSimilarityRanker,
)

from retrieval_components.dataclasses.query import Query
from retrieval_components.models.sentence_transformers_similarity_ranker import (
    SentenceTransformersSimilarityRanker,
)
from retrieval_components.models.sentence_transformers_text_embedder import (
    SentenceTransformersTextEmbedder,
)
from retrieval_components.models.transformers_similarity_ranker import TransformersSimilarityRanker


def test_sentence_transformers_text_embedder_accepts_query(monkeypatch) -> None:
    received: list[str] = []

    def fake_run(
        self: HaystackSentenceTransformersTextEmbedder,
        text: str,
    ) -> dict[str, Any]:
        received.append(text)
        return {"embedding": [1.0, 2.0]}

    monkeypatch.setattr(HaystackSentenceTransformersTextEmbedder, "run", fake_run)
    embedder = SentenceTransformersTextEmbedder(progress_bar=False)

    with pytest.raises(ValueError, match="no materialized content"):
        embedder.run(Query(id="q1"))
    assert embedder.run(Query(id="q1", content="query text")) == {"embedding": [1.0, 2.0]}
    assert received == ["query text"]
    assert embedder.__haystack_input__.get("query").type is Query
    assert embedder.to_dict()["type"].endswith(
        "sentence_transformers_text_embedder.SentenceTransformersTextEmbedder"
    )


def test_sentence_transformers_similarity_ranker_accepts_query(monkeypatch) -> None:
    documents = [Document(id="d1", content="document")]
    received: list[str] = []

    def fake_run(
        self: HaystackSentenceTransformersSimilarityRanker,
        *,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        scale_score: bool | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, list[Document]]:
        received.append(query)
        return {"documents": documents}

    monkeypatch.setattr(HaystackSentenceTransformersSimilarityRanker, "run", fake_run)
    ranker = SentenceTransformersSimilarityRanker()

    with pytest.raises(ValueError, match="no materialized content"):
        ranker.run(query=Query(id="q1"), documents=[])
    assert ranker.run(query=Query(id="q1", content="query text"), documents=documents) == {
        "documents": documents
    }
    assert received == ["query text"]
    assert ranker.__haystack_input__.get("query").type is Query


def test_transformers_similarity_ranker_accepts_query(monkeypatch) -> None:
    documents = [Document(id="d1", content="document")]
    received: list[str] = []

    def fake_run(
        self: HaystackTransformersSimilarityRanker,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        scale_score: bool | None = None,
        calibration_factor: float | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        received.append(query)
        return {"documents": documents}

    monkeypatch.setattr(
        "haystack.components.rankers.transformers_similarity.torch_and_transformers_import.check",
        lambda: None,
    )
    monkeypatch.setattr(HaystackTransformersSimilarityRanker, "run", fake_run)
    ranker = TransformersSimilarityRanker()

    with pytest.raises(ValueError, match="no materialized content"):
        ranker.run(query=Query(id="q1"), documents=[])
    assert ranker.run(Query(id="q1", content="query text"), documents) == {"documents": documents}
    assert received == ["query text"]
    assert ranker.__haystack_input__.get("query").type is Query
