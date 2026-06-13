from retrieval_research.config import compose_stage_config
from retrieval_research.pipelines import load_async_pipeline


def test_rrf_fusion_pipeline_config_loads_with_dynamic_weight_socket() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "pipeline/inference@pipeline=rrf_fusion",
            "pipeline.components.fusion.init_parameters.weights={lexical:1.0}",
        ],
    )

    pipeline = load_async_pipeline(cfg.pipeline)

    assert "fusion" in pipeline.graph.nodes
