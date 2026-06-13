from haystack import Document
from omegaconf import OmegaConf

from retrieval_research.components import ReciprocalRankFusion, ScoreFusion, WeightedDocumentFusion
from retrieval_research.components.cascade import TopKDocuments, TopPDocuments
from retrieval_research.stages.inference import _build_query_inputs


def test_weighted_document_fusion_uses_dynamic_sources() -> None:
    fusion = WeightedDocumentFusion(weights={"lexical": 1.0, "dense": 2.0}, top_k=2, rrf_k=10)

    result = fusion.run(
        lexical=[
            Document(id="d1", content="first"),
            Document(id="d2", content="second"),
        ],
        dense=[
            Document(id="d2", content="second"),
            Document(id="d3", content="third"),
        ],
    )

    assert [document.id for document in result["documents"]] == ["d2", "d3"]
    assert result["documents"][0].score > result["documents"][1].score


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


def test_build_query_inputs_supports_multiple_query_targets() -> None:
    cfg = OmegaConf.create(
        {
            "pipeline_run": {
                "inputs": {},
                "query_inputs": [
                    {"component": "lexical", "parameter": "query"},
                    {"component": "dense", "parameter": "query"},
                ],
            },
        }
    )

    assert _build_query_inputs(cfg, "hydra pipelines") == {
        "lexical": {"query": "hydra pipelines"},
        "dense": {"query": "hydra pipelines"},
    }
