from omegaconf import OmegaConf

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.pipelines import load_async_pipeline, to_container


def test_rrf_fusion_pipeline_config_loads_with_dynamic_weight_socket() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=rrf_fusion",
            "pipeline.components.fusion.init_parameters.weights={lexical:1.0}",
        ],
    )

    pipeline = load_async_pipeline(cfg.pipeline)
    pipeline_config = to_container(cfg.pipeline)

    assert "fusion" in pipeline.graph.nodes
    assert pipeline_config["components"]["fusion"]["init_parameters"]["rrf_k"] == 60
    assert "retrieval" not in cfg


def test_abstract_dense_e5_indexing_config_keeps_pipeline_haystack_shaped() -> None:
    cfg = compose_stage_config(
        "indexing",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/indexing@pipeline=dense_jsonl",
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
    document_source_parameters = pipeline_config["components"]["document_source"]["init_parameters"]
    assert document_source_parameters["id_field"] == EVALUATION_DATA_SCHEMA.doc_id
    assert document_source_parameters["content_field"] is None
    assert (
        pipeline_config["components"]["document_parser"]["init_parameters"]["content_field"]
        == EVALUATION_DATA_SCHEMA.text
    )
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
    assert "embedder" in pipeline.graph.nodes


def test_abstract_chunked_e5_configs_use_chunked_index_artifact() -> None:
    indexing_cfg = compose_stage_config(
        "indexing",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/indexing@pipeline=dense_chunked_jsonl",
            "selections.index_id=toy-dense-chunked",
            "selections/embedding_model=e5/small_v2",
        ],
    )
    inference_cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=dense_chunked_jsonl",
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
            "pipeline/inference@pipeline=dense_jsonl",
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
    assert pipeline_config["components"]["query_embedder"]["init_parameters"]["device"] == {
        "type": "single",
        "device": "cuda",
    }
    assert (
        pipeline_config["components"]["query_embedder"]["init_parameters"]["progress_bar"] is True
    )
    assert pipeline_config["components"]["retriever"]["init_parameters"]["similarity"] == "cosine"
    assert {"sender": "input.query_meta", "receiver": "query_parser.meta"} in pipeline_config[
        "connections"
    ]
    assert {"sender": "query_parser.text", "receiver": "query_preprocessor.text"} in (
        pipeline_config["connections"]
    )
    assert {"sender": "query_parser.text", "receiver": "output.query_content"} in (
        pipeline_config["connections"]
    )
    assert {
        "sender": "input.candidate_document_ids",
        "receiver": "retriever.candidate_document_ids",
    } in pipeline_config["connections"]
    assert {"sender": "retriever.documents", "receiver": "output.documents"} in pipeline_config[
        "connections"
    ]


def test_dense_candidate_reranker_uses_candidate_documents() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=dense_candidate_reranker",
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


def test_cross_encoder_candidate_reranker_uses_bge_selection() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=cross_encoder_candidate_reranker",
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
    assert pipeline_config["components"]["ranker"]["init_parameters"]["scale_score"] is True
    assert {
        "sender": "input.candidate_documents",
        "receiver": "document_parser.documents",
    } in pipeline_config["connections"]
    assert {
        "sender": "document_parser.documents",
        "receiver": "ranker.documents",
    } in pipeline_config["connections"]
    assert {"sender": "ranker.documents", "receiver": "output.documents"} in pipeline_config[
        "connections"
    ]


def test_dummy_keyword_inference_parses_retriever_query_input() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=dummy_keyword",
            "selections.index_id=toy-keyword",
        ],
    )
    pipeline_config = to_container(cfg.pipeline)

    assert {"sender": "input.query_meta", "receiver": "query_parser.meta"} in pipeline_config[
        "connections"
    ]
    assert {"sender": "query_parser.text", "receiver": "retriever.query"} in pipeline_config[
        "connections"
    ]
    assert {
        "sender": "input.candidate_document_ids",
        "receiver": "retriever.candidate_document_ids",
    } in pipeline_config["connections"]
    assert {"sender": "retriever.documents", "receiver": "output.documents"} in pipeline_config[
        "connections"
    ]
