from pathlib import Path

from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.pipelines import load_async_pipeline, to_container

PROJECT_DIR = Path(__file__).parents[1]


def test_project_owned_config_choices_are_namespaced() -> None:
    config_files = sorted((PROJECT_DIR / "configs").rglob("*.yaml"))

    assert config_files
    assert all(
        path.relative_to(PROJECT_DIR / "configs").parts[2]
        == "sparse_autoencoder_retrieval"
        for path in config_files
    )


def _compose(stage: str):
    return compose_stage_config(
        stage,
        [
            "dataset=toy",
            "runtime=cpu",
            f"pipeline/{stage}@pipeline=sparse_autoencoder_retrieval/semantic_sparse",
            "selections/embedding_model=e5/small_v2",
            (
                "selections/sparse_autoencoder="
                "sparse_autoencoder_retrieval/e5_small_ccsa"
            ),
            "selections.index_id=toy-e5-small-ccsa",
            f"stage.run_id=toy-ccsa-{stage}",
        ],
        project_dir=PROJECT_DIR,
    )


def test_semantic_sparse_indexing_pipeline_loads() -> None:
    cfg = _compose("indexing")
    pipeline_dict = to_container(cfg.pipeline)
    pipeline = load_async_pipeline(cfg.pipeline)

    assert {
        "input",
        "document_parser",
        "document_prefixer",
        "embedder",
        "indexer",
        "output",
    } <= set(pipeline.graph.nodes)
    assert pipeline_dict["components"]["embedder"]["init_parameters"]["model"] == (
        "intfloat/e5-small-v2"
    )
    assert pipeline_dict["components"]["indexer"]["type"].endswith("SemanticSparseIndexer")


def test_semantic_sparse_inference_pipeline_loads() -> None:
    cfg = _compose("inference")
    pipeline_dict = to_container(cfg.pipeline)
    pipeline = load_async_pipeline(cfg.pipeline)

    assert {
        "input",
        "query_preprocessor",
        "query_to_string",
        "query_embedder",
        "retriever",
        "output",
    } <= set(pipeline.graph.nodes)
    assert "query_parser" not in pipeline.graph.nodes
    assert pipeline_dict["components"]["query_embedder"]["type"].endswith(
        "SparseAutoencoderTextEmbedder"
    )
    assert {
        "sender": "query_embedder.sparse_embedding",
        "receiver": "retriever.sparse_embedding",
    } in pipeline_dict["connections"]
