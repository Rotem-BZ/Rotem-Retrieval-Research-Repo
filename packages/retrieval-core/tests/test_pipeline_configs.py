from pathlib import Path

from omegaconf import OmegaConf

from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.pipelines import (
    load_async_pipeline,
    to_container,
    without_component_progress_bars,
)


def test_stage_pipeline_copy_disables_component_progress_bars() -> None:
    pipeline_config = {
        "components": {
            "first": {"init_parameters": {"progress_bar": True}},
            "second": {"init_parameters": {"show_progress_bar": True}},
            "third": {"init_parameters": {"unrelated": True}},
        }
    }

    stage_config = without_component_progress_bars(pipeline_config)

    assert stage_config["components"]["first"]["init_parameters"]["progress_bar"] is False
    assert (
        stage_config["components"]["second"]["init_parameters"]["show_progress_bar"]
        is False
    )
    assert stage_config["components"]["third"]["init_parameters"] == {"unrelated": True}
    assert pipeline_config["components"]["first"]["init_parameters"]["progress_bar"] is True


def test_abstract_dense_e5_indexing_config_keeps_pipeline_haystack_shaped() -> None:
    cfg = compose_stage_config(
        "indexing",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/indexing@pipeline=dense/documents_in_memory",
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
    assert pipeline_config["components"]["indexer"]["init_parameters"]["similarity"] == "cosine"
    assert pipeline_config["components"]["indexer"]["type"].endswith(
        "persisted_in_memory_document_indexer.PersistedInMemoryDocumentIndexer"
    )
    assert {"input", "embedder", "output"} <= set(pipeline.graph.nodes)


def test_chunked_indexing_and_dense_retrieval_share_index_artifact() -> None:
    indexing_cfg = compose_stage_config(
        "indexing",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/indexing@pipeline=dense/chunks_in_memory",
            "selections.index_id=toy-dense-chunked",
            "selections/embedding_model=e5/small_v2",
        ],
    )
    inference_cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=retrieve/dense_in_memory",
            "selections.index_id=toy-dense-chunked",
            "selections/embedding_model=e5/small_v2",
        ],
    )

    indexing_pipeline_config = to_container(indexing_cfg.pipeline)
    inference_pipeline_config = to_container(inference_cfg.pipeline)

    assert "splitter" in indexing_pipeline_config["components"]
    assert indexing_pipeline_config["components"]["indexer"]["init_parameters"][
        "output_path"
    ].endswith("/index.json")
    assert inference_pipeline_config["components"]["retriever"]["init_parameters"][
        "index_path"
    ].endswith("artifacts/indexes/toy-dense-chunked/index.json")


def test_abstract_dense_e5_inference_config_prefixes_queries() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=gpu",
            "pipeline/inference@pipeline=retrieve/dense_in_memory",
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
    assert pipeline_config["components"]["retriever"]["type"].endswith(
        "persisted_in_memory_embedding_retriever.PersistedInMemoryEmbeddingRetriever"
    )
    assert {"sender": "input.query", "receiver": "query_preprocessor.query"} in pipeline_config[
        "connections"
    ]
    assert {"sender": "input.query", "receiver": "output.query"} in (pipeline_config["connections"])
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


def test_cross_encoder_reranker_uses_bge_selection_and_prefix_components() -> None:
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
    assert pipeline_config["components"]["query_preprocessor"]["init_parameters"]["prefix"] == ""
    assert pipeline_config["components"]["document_prefixer"]["init_parameters"]["prefix"] == ""
    assert {"query_preprocessor", "document_prefixer"} <= set(pipeline.graph.nodes)
    assert {
        "sender": "input.candidate_documents",
        "receiver": "document_parser.documents",
    } in pipeline_config["connections"]
    assert {
        "sender": "document_parser.documents",
        "receiver": "document_prefixer.documents",
    } in pipeline_config["connections"]
    assert {
        "sender": "query_preprocessor.query",
        "receiver": "ranker.query",
    } in pipeline_config["connections"]
    assert {
        "sender": "document_prefixer.documents",
        "receiver": "ranker.documents",
    } in pipeline_config["connections"]
    assert {"sender": "ranker.documents", "receiver": "output.documents"} in pipeline_config[
        "connections"
    ]


def test_cross_encoder_reranker_binds_nonempty_model_prefixes(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    model_path = config_dir / "selections" / "reranker_model" / "test" / "prefixed.yaml"
    model_path.parent.mkdir(parents=True)
    model_path.write_text(
        """name: test_prefixed_reranker
artifact_name: test_prefixed_reranker
checkpoint: test/prefixed-reranker
document_prefix: "rerank passage: "
query_prefix: "rerank query: "
scale_score: false
tokenizer_kwargs:
  model_max_length: 256
""",
        encoding="utf-8",
    )

    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=cpu",
            "pipeline/inference@pipeline=rerank/cross_encoder",
            "selections/reranker_model=test/prefixed",
        ],
        config_dir=config_dir,
    )

    pipeline_config = to_container(cfg.pipeline)
    load_async_pipeline(cfg.pipeline)

    assert (
        pipeline_config["components"]["query_preprocessor"]["init_parameters"]["prefix"]
        == "rerank query: "
    )
    assert (
        pipeline_config["components"]["document_prefixer"]["init_parameters"]["prefix"]
        == "rerank passage: "
    )
    assert pipeline_config["components"]["ranker"]["init_parameters"]["model"] == (
        "test/prefixed-reranker"
    )


def test_embedding_model_catalog_can_fill_two_named_model_roles(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    pipeline_path = config_dir / "pipeline" / "inference" / "test" / "two_embedding_models.yaml"
    reranker_model_path = config_dir / "selections" / "embedding_model" / "test" / "reranker.yaml"
    pipeline_path.parent.mkdir(parents=True)
    reranker_model_path.parent.mkdir(parents=True)
    reranker_model_path.write_text(
        """name: test_reranker
artifact_name: test_reranker
checkpoint: test/reranker
document_prefix: "rerank passage: "
query_prefix: "rerank query: "
normalize_embeddings: false
similarity: dot_product
tokenizer_kwargs:
  model_max_length: 256
""",
        encoding="utf-8",
    )
    pipeline_path.write_text(
        """defaults:
  - /selections/embedding_model@_global_.selections.models.retriever: ???
  - /selections/embedding_model@_global_.selections.models.reranker: ???
  - /component/query_preprocessor@components.retriever_query_preprocessor: prefix_cleanup
  - /component/query_embedder@components.retriever_query_embedder: sentence_transformers
  - /component/query_preprocessor@components.reranker_query_preprocessor: prefix_cleanup
  - /component/query_embedder@components.reranker_query_embedder: sentence_transformers
  - _self_

components:
  input:
    type: retrieval_components.interfaces.inference.InferenceInput
  retriever_query_preprocessor:
    init_parameters:
      prefix: ${selections.models.retriever.query_prefix}
  retriever_query_embedder:
    init_parameters:
      model: ${selections.models.retriever.checkpoint}
      normalize_embeddings: ${selections.models.retriever.normalize_embeddings}
      tokenizer_kwargs: ${selections.models.retriever.tokenizer_kwargs}
  reranker_query_preprocessor:
    init_parameters:
      prefix: ${selections.models.reranker.query_prefix}
  reranker_query_embedder:
    init_parameters:
      model: ${selections.models.reranker.checkpoint}
      normalize_embeddings: ${selections.models.reranker.normalize_embeddings}
      tokenizer_kwargs: ${selections.models.reranker.tokenizer_kwargs}
  output:
    type: retrieval_components.interfaces.inference.InferenceOutput

connections:
  - sender: input.query
    receiver: retriever_query_preprocessor.query
  - sender: input.query
    receiver: reranker_query_preprocessor.query
  - sender: input.query
    receiver: output.query
  - sender: retriever_query_preprocessor.query
    receiver: retriever_query_embedder.query
  - sender: reranker_query_preprocessor.query
    receiver: reranker_query_embedder.query
  - sender: input.candidate_documents
    receiver: output.documents

max_runs_per_component: 100
metadata:
  description: Test-only topology with two model roles from one catalog.
""",
        encoding="utf-8",
    )

    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=cpu",
            "pipeline/inference@pipeline=test/two_embedding_models",
            ("selections/embedding_model@selections.models.retriever=e5/small_v2"),
            ("selections/embedding_model@selections.models.reranker=test/reranker"),
        ],
        config_dir=config_dir,
    )

    pipeline_config = to_container(cfg.pipeline)
    pipeline = load_async_pipeline(cfg.pipeline)

    assert cfg.selections.models.retriever.checkpoint == "intfloat/e5-small-v2"
    assert cfg.selections.models.reranker.checkpoint == "test/reranker"
    assert (
        pipeline_config["components"]["retriever_query_preprocessor"]["init_parameters"]["prefix"]
        == "query: "
    )
    assert (
        pipeline_config["components"]["reranker_query_preprocessor"]["init_parameters"]["prefix"]
        == "rerank query: "
    )
    assert (
        pipeline_config["components"]["retriever_query_embedder"]["init_parameters"]["model"]
        == "intfloat/e5-small-v2"
    )
    assert (
        pipeline_config["components"]["reranker_query_embedder"]["init_parameters"]["model"]
        == "test/reranker"
    )
    assert {"retriever_query_embedder", "reranker_query_embedder"} <= set(pipeline.graph.nodes)


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
