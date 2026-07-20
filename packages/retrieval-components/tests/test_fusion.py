import pytest
from haystack import Document

from retrieval_components import (
    LinearScoreFusion,
    ReciprocalRankFusion,
    ScoreFusion,
    ZScoreFusion,
)
from retrieval_components.cascade import ChunkCascade, TopKDocuments, TopPDocuments


def test_reciprocal_rank_fusion_uses_named_source_weights() -> None:
    fusion = ReciprocalRankFusion(weights={"lexical": 1.0, "dense": 2.0}, top_k=2, rrf_k=10)

    result = fusion.run(
        lexical=[Document(id="d1"), Document(id="d2")],
        dense=[Document(id="d2"), Document(id="d3")],
    )

    assert [document.id for document in result["documents"]] == ["d2", "d3"]


def test_score_fusion_sums_weighted_scores() -> None:
    fusion = ScoreFusion(weights={"lexical": 1.0, "dense": 2.0})

    result = fusion.run(
        lexical=[Document(id="d1", score=0.4), Document(id="d2", score=0.3)],
        dense=[Document(id="d1", score=0.2), Document(id="d3", score=0.9)],
    )

    assert [document.id for document in result["documents"]] == ["d3", "d1", "d2"]
    assert result["documents"][1].score == 0.8


def test_score_fusion_preserves_optional_linear_normalization() -> None:
    fusion = ScoreFusion(weights={"source": 1.0}, normalize_by_source=True)

    result = fusion.run(source=[Document(id="low", score=10.0), Document(id="high", score=20.0)])

    assert [document.id for document in result["documents"]] == ["high", "low"]
    assert [document.score for document in result["documents"]] == [1.0, 0.0]


def test_z_score_fusion_normalizes_each_source() -> None:
    fusion = ZScoreFusion(weights={"lexical": 1.0, "dense": 1.0})

    result = fusion.run(
        lexical=[Document(id="d1", score=10.0), Document(id="d2", score=20.0)],
        dense=[Document(id="d1", score=100.0), Document(id="d3", score=0.0)],
    )

    assert [document.id for document in result["documents"]] == ["d2", "d1", "d3"]
    scores = {document.id: document.score for document in result["documents"]}
    assert scores == pytest.approx({"d1": 0.0, "d2": 1.0, "d3": -1.0})


def test_linear_score_fusion_normalizes_each_source() -> None:
    fusion = LinearScoreFusion(weights={"source": 1.0})

    result = fusion.run(source=[Document(id="low", score=10.0), Document(id="high", score=20.0)])

    assert [document.id for document in result["documents"]] == ["high", "low"]
    assert [document.score for document in result["documents"]] == [1.0, 0.0]


def test_z_score_fusion_normalizes_constant_source_to_zero() -> None:
    fusion = ZScoreFusion(weights={"only": 2.0})

    result = fusion.run(only=[Document(id="d1", score=3.0), Document(id="d2", score=3.0)])

    assert [document.score for document in result["documents"]] == [0.0, 0.0]


def test_top_k_and_top_p_cascade_selectors() -> None:
    documents = [
        Document(id="d1", score=0.6),
        Document(id="d2", score=0.3),
        Document(id="d3", score=0.1),
    ]

    assert [document.id for document in TopKDocuments(top_k=2).run(documents)["documents"]] == [
        "d1",
        "d2",
    ]
    assert [document.id for document in TopPDocuments(top_p=0.8).run(documents)["documents"]] == [
        "d1",
        "d2",
    ]


def test_chunk_cascade_keeps_top_k_chunks_per_source_document() -> None:
    documents = [
        Document(id="d1::0", score=0.9, meta={"source_document_id": "d1"}),
        Document(id="d2::0", score=0.8, meta={"source_document_id": "d2"}),
        Document(id="d1::1", score=0.7, meta={"source_document_id": "d1"}),
        Document(id="d2::1", score=0.6, meta={"source_document_id": "d2"}),
        Document(id="d1::2", score=0.5, meta={"source_document_id": "d1"}),
    ]

    result = ChunkCascade(top_k=2).run(documents)

    assert [document.id for document in result["documents"]] == [
        "d1::0",
        "d2::0",
        "d1::1",
        "d2::1",
    ]
    assert result["documents"][0] is documents[0]


def test_chunk_cascade_always_selects_by_score() -> None:
    documents = [
        Document(id="first", score=0.1, meta={"source_document_id": "d1"}),
        Document(id="second", score=0.9, meta={"source_document_id": "d1"}),
    ]

    result = ChunkCascade(top_k=1).run(documents)

    assert [document.id for document in result["documents"]] == ["second"]


def test_chunk_cascade_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        ChunkCascade(top_k=0)
