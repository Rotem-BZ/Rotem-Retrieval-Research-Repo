from haystack import Document

from retrieval_research.components import ReciprocalRankFusion, ScoreFusion
from retrieval_research.components.cascade import TopKDocuments, TopPDocuments
from retrieval_research.stages.inference import _build_query_inputs


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


def test_build_query_inputs_targets_fixed_inference_input_component() -> None:
    documents = [Document(id="d1", content="one")]

    assert _build_query_inputs(
        "hydra pipelines",
        candidate_document_ids=["d1"],
        candidate_documents=documents,
    ) == {
        "input": {
            "query": "hydra pipelines",
            "candidate_document_ids": ["d1"],
            "candidate_documents": documents,
        }
    }
