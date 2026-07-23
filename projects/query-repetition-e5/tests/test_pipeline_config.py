from pathlib import Path

from retrieval_core.utils.config import compose_entrypoint_config, compose_stage_config
from retrieval_core.utils.pipelines import load_async_pipeline


EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments" / "query-repetition-e5-small-scifact"
PROJECT_DIR = Path(__file__).parents[1]
TOY_EXPERIMENT_DIR = (
    Path(__file__).parents[1] / "experiments" / "query-repetition-e5-small-toy"
)


def test_project_owned_config_choices_are_namespaced() -> None:
    config_files = sorted((PROJECT_DIR / "configs").rglob("*.yaml"))

    assert config_files
    assert all(
        path.relative_to(PROJECT_DIR / "configs").parts[2] == "query_repetition_e5"
        for path in config_files
    )


def test_project_pipeline_composes_with_core_config_groups() -> None:
    cfg = compose_entrypoint_config(
        EXPERIMENT_DIR / "configs" / "runs" / "repeated.yaml",
    )

    assert cfg.pipeline.components.query_repeater.type.endswith("QueryRepeater")
    assert cfg.stage.run_id.endswith("--repeated")
    load_async_pipeline(cfg.pipeline)


def test_toy_experiment_run_uses_project_component_and_repository_fixture() -> None:
    cfg = compose_entrypoint_config(
        TOY_EXPERIMENT_DIR / "configs" / "runs" / "repeated.yaml",
    )

    repository_root = Path(__file__).parents[3]
    assert cfg.dataset.name == "toy"
    assert Path(cfg.dataset.documents_path).resolve() == (
        repository_root / "data" / "processed" / "toy" / "documents.jsonl"
    ).resolve()
    assert cfg.selections.index_id == "toy-e5-small-index"
    assert cfg.pipeline.components.query_repeater.type.endswith("QueryRepeater")
    assert cfg.stage.run_id == "query-repetition-e5-small-toy--repeated"
    load_async_pipeline(cfg.pipeline)


def test_documented_direct_toy_component_command_composes() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "paths.processed_data_dir=../../data/processed",
            "runtime=cpu",
            "pipeline/inference@pipeline=query_repetition_e5/dense_query_repetition",
            "selections/embedding_model=e5/small_v2",
            "selections.index_id=toy-e5-small-index",
            "stage.run_id=toy-query-repetition-component",
        ],
        project_dir=PROJECT_DIR,
    )

    assert cfg.dataset.name == "toy"
    assert cfg.pipeline.components.query_repeater.type.endswith("QueryRepeater")
    assert cfg.selections.index_id == "toy-e5-small-index"
    assert cfg.stage.run_id == "toy-query-repetition-component"
    load_async_pipeline(cfg.pipeline)
