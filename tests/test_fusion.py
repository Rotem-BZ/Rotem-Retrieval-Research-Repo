from haystack import Document
from omegaconf import OmegaConf

from retrieval_research.components import WeightedDocumentFusion
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
