from retrieval_research.config import compose_stage_config
from retrieval_research.pipelines import to_container
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


def test_abstract_dense_e5_indexing_config_keeps_pipeline_haystack_shaped() -> None:
    cfg = compose_stage_config(
        "indexing",
        [
            "dataset=toy",
            "pipeline/indexing@pipeline=dense_jsonl",
            "choices/embedding_model=e5/small_v2",
        ],
    )

    pipeline_config = to_container(cfg.pipeline)
    pipeline = load_async_pipeline(cfg.pipeline)

    assert set(pipeline_config) == {
        "components",
        "connections",
        "max_runs_per_component",
        "metadata",
    }
    assert cfg.choices.embedding_model.checkpoint == "intfloat/e5-small-v2"
    assert pipeline_config["components"]["document_prefixer"]["init_parameters"]["prefix"] == "passage: "
    assert pipeline_config["components"]["embedder"]["init_parameters"]["model"] == "intfloat/e5-small-v2"
    assert "embedder" in pipeline.graph.nodes


def test_abstract_chunked_e5_configs_use_chunked_index_artifact() -> None:
    indexing_cfg = compose_stage_config(
        "indexing",
        [
            "dataset=toy",
            "pipeline/indexing@pipeline=dense_chunked_jsonl",
            "choices/embedding_model=e5/small_v2",
        ],
    )
    inference_cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "pipeline/inference@pipeline=dense_chunked_jsonl",
            "choices/embedding_model=e5/small_v2",
        ],
    )

    indexing_pipeline_config = to_container(indexing_cfg.pipeline)
    inference_pipeline_config = to_container(inference_cfg.pipeline)

    assert "splitter" in indexing_pipeline_config["components"]
    assert indexing_pipeline_config["components"]["indexer"]["init_parameters"]["output_path"].endswith(
        "toy_e5_small_v2_chunked.jsonl"
    )
    assert inference_pipeline_config["components"]["retriever"]["init_parameters"]["index_path"].endswith(
        "toy_e5_small_v2_chunked.jsonl"
    )


def test_abstract_dense_e5_inference_config_prefixes_queries() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "pipeline/inference@pipeline=dense_jsonl",
            "choices/embedding_model=e5/small_v2",
        ],
    )

    pipeline_config = to_container(cfg.pipeline)

    assert pipeline_config["components"]["query_preprocessor"]["init_parameters"]["prefix"] == "query: "
    assert pipeline_config["components"]["query_embedder"]["init_parameters"]["model"] == "intfloat/e5-small-v2"
    assert pipeline_config["components"]["retriever"]["init_parameters"]["similarity"] == "cosine"
