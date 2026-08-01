from pathlib import Path

from omegaconf import OmegaConf

from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.pipelines import load_async_pipeline, to_container


def test_abstract_dense_e5_indexing_config_keeps_pipeline_haystack_shaped() -> None:
    cfg = compose_stage_config(
        "indexing",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/indexing@pipeline=dense/documents_jsonl",
            "selections.index_id=toy-dense",
            "selections/embedding_model=e5/small_v2",
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
    assert cfg.selections.embedding_model.checkpoint == "intfloat/e5-small-v2"
    assert "document_source" not in pipeline_config["components"]
    assert pipeline_config["components"]["input"]["type"].endswith("IndexingInput")
    assert pipeline_config["components"]["document_parser"]["type"].endswith(
        "identity_parser.IdentityParser"
    )
    assert {
        "sender": "input.documents",
        "receiver": "document_parser.documents",
    } in pipeline_config["connections"]
    assert (
        pipeline_config["components"]["document_prefixer"]["init_parameters"]["prefix"]
        == "passage: "
    )
    assert (
        pipeline_config["components"]["embedder"]["init_parameters"]["model"]
        == "intfloat/e5-small-v2"
    )
    assert pipeline_config["components"]["embedder"]["init_parameters"]["device"] == {
        "type": "single",
        "device": "cuda",
    }
    assert pipeline_config["components"]["embedder"]["init_parameters"]["progress_bar"] is True
    assert {"input", "embedder", "output"} <= set(pipeline.graph.nodes)


def test_chunked_indexing_and_dense_retrieval_share_index_artifact() -> None:
    indexing_cfg = compose_stage_config(
        "indexing",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/indexing@pipeline=dense/chunks_jsonl",
            "selections.index_id=toy-dense-chunked",
            "selections/embedding_model=e5/small_v2",
        ],
    )
    inference_cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=retrieve/dense_jsonl",
            "selections.index_id=toy-dense-chunked",
            "selections/embedding_model=e5/small_v2",
        ],
    )

    indexing_pipeline_config = to_container(indexing_cfg.pipeline)
    inference_pipeline_config = to_container(inference_cfg.pipeline)

    assert "splitter" in indexing_pipeline_config["components"]
    assert indexing_pipeline_config["components"]["indexer"]["init_parameters"][
        "output_path"
    ].endswith("/index.jsonl")
    assert inference_pipeline_config["components"]["retriever"]["init_parameters"][
        "index_path"
    ].endswith("artifacts/indexes/toy-dense-chunked/index.jsonl")


def test_abstract_dense_e5_inference_config_prefixes_queries() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=retrieve/dense_jsonl",
            "selections.index_id=toy-dense",
            "selections/embedding_model=e5/small_v2",
        ],
    )

    pipeline_config = to_container(cfg.pipeline)

    assert cfg.selections.index_id == "toy-dense"
    assert not OmegaConf.is_missing(
        cfg.pipeline.components.retriever.init_parameters,
        "index_path",
    )
    assert (
        pipeline_config["components"]["query_preprocessor"]["init_parameters"]["prefix"]
        == "query: "
    )
    assert (
        pipeline_config["components"]["query_embedder"]["init_parameters"]["model"]
        == "intfloat/e5-small-v2"
    )
    assert pipeline_config["components"]["query_embedder"]["type"].endswith(
        "retrieval_components.models.sentence_transformers_text_embedder."
        "SentenceTransformersTextEmbedder"
    )
    assert pipeline_config["components"]["query_embedder"]["init_parameters"]["device"] == {
        "type": "single",
        "device": "cuda",
    }
    assert (
        pipeline_config["components"]["query_embedder"]["init_parameters"]["progress_bar"] is True
    )
    assert pipeline_config["components"]["retriever"]["init_parameters"]["similarity"] == "cosine"
    assert {"sender": "input.query", "receiver": "query_preprocessor.query"} in pipeline_config[
        "connections"
    ]
    assert {"sender": "input.query", "receiver": "output.query"} in (
        pipeline_config["connections"]
    )
    assert "query_parser" not in pipeline_config["components"]
    assert {"sender": "query_preprocessor.query", "receiver": "query_embedder.query"} in (
        pipeline_config["connections"]
    )
    assert "query_adapter" not in pipeline_config["components"]
    assert {
        "sender": "input.candidate_document_ids",
        "receiver": "retriever.candidate_document_ids",
    } in pipeline_config["connections"]
    assert {"sender": "retriever.documents", "receiver": "output.documents"} in pipeline_config[
        "connections"
    ]


def test_bi_encoder_reranker_uses_candidate_documents() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=rerank/bi_encoder",
            "selections/embedding_model=e5/small_v2",
        ],
    )

    pipeline_config = to_container(cfg.pipeline)
    pipeline = load_async_pipeline(cfg.pipeline)

    assert "index_path" not in cfg.pipeline.components.ranker.init_parameters
    assert "index_id" not in cfg.selections
    assert "ranker" in pipeline.graph.nodes
    assert pipeline_config["components"]["ranker"]["init_parameters"]["similarity"] == "cosine"
    assert {
        "sender": "input.candidate_documents",
        "receiver": "document_parser.documents",
    } in pipeline_config["connections"]
    assert {
        "sender": "document_parser.documents",
        "receiver": "document_prefixer.documents",
    } in pipeline_config["connections"]
    assert {
        "sender": "query_embedder.embedding",
        "receiver": "ranker.query_embedding",
    } in pipeline_config["connections"]
    assert {"sender": "ranker.documents", "receiver": "output.documents"} in pipeline_config[
        "connections"
    ]


def test_cross_encoder_reranker_uses_bge_selection() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=rerank/cross_encoder",
            "selections/reranker_model=bge/v2_m3",
        ],
    )

    pipeline_config = to_container(cfg.pipeline)
    pipeline = load_async_pipeline(cfg.pipeline)

    assert "index_path" not in cfg.pipeline.components.ranker.init_parameters
    assert "index_id" not in cfg.selections
    assert cfg.selections.reranker_model.checkpoint == "BAAI/bge-reranker-v2-m3"
    assert "ranker" in pipeline.graph.nodes
    assert (
        pipeline_config["components"]["ranker"]["init_parameters"]["model"]
        == "BAAI/bge-reranker-v2-m3"
    )
    assert pipeline_config["components"]["ranker"]["type"].endswith(
        "retrieval_components.models.sentence_transformers_similarity_ranker."
        "SentenceTransformersSimilarityRanker"
    )
    assert pipeline_config["components"]["ranker"]["init_parameters"]["scale_score"] is True
    assert {
        "sender": "input.candidate_documents",
        "receiver": "document_parser.documents",
    } in pipeline_config["connections"]
    assert {
        "sender": "document_parser.documents",
        "receiver": "ranker.documents",
    } in pipeline_config["connections"]
    assert {
        "sender": "input.query",
        "receiver": "ranker.query",
    } in pipeline_config["connections"]
    assert {"sender": "ranker.documents", "receiver": "output.documents"} in pipeline_config[
        "connections"
    ]


def test_materialized_native_haystack_pipeline_keeps_query_to_string_adapter() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "src"
        / "retrieval_core"
        / "configs"
        / "materialized"
        / "production"
        / "toy_dense_inference_native_adapter_reference.yaml"
    )
    cfg = OmegaConf.load(config_path)
    pipeline_config = to_container(cfg.pipeline)
    pipeline = load_async_pipeline(cfg.pipeline)

    assert {"query_to_string", "query_embedder"} <= set(pipeline.graph.nodes)
    assert pipeline_config["components"]["query_to_string"]["type"].endswith(
        "query_to_string.QueryToString"
    )
    assert pipeline_config["components"]["query_embedder"]["type"].startswith(
        "haystack.components.embedders."
    )
    assert {
        "sender": "query_preprocessor.query",
        "receiver": "query_to_string.query",
    } in pipeline_config["connections"]
    assert {
        "sender": "query_to_string.text",
        "receiver": "query_embedder.text",
    } in pipeline_config["connections"]
