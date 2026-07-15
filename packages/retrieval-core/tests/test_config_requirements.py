import pytest
from hydra.errors import ConfigCompositionException

from retrieval_core.config import compose_stage_config
from retrieval_core.stages.base import prepare_stage_run_config


def test_indexing_requires_dataset_and_pipeline_selection() -> None:
    with pytest.raises(ConfigCompositionException, match="pipeline/indexing@pipeline"):
        compose_stage_config("indexing")

    with pytest.raises(ConfigCompositionException, match="dataset"):
        compose_stage_config("indexing", ["pipeline/indexing@pipeline=dummy_jsonl"])


def test_inference_requires_dataset_and_pipeline_selection() -> None:
    with pytest.raises(ConfigCompositionException, match="pipeline/inference@pipeline"):
        compose_stage_config("inference", ["dataset=toy"])

    with pytest.raises(ConfigCompositionException, match="dataset"):
        compose_stage_config("inference", ["pipeline/inference@pipeline=dummy_keyword"])


def test_evaluation_requires_dataset_selection() -> None:
    with pytest.raises(ConfigCompositionException, match="dataset"):
        compose_stage_config("evaluation")


def test_explicit_stage_selections_compose() -> None:
    indexing_cfg = compose_stage_config(
        "indexing",
        ["dataset=toy", "pipeline/indexing@pipeline=dummy_jsonl", "stage.run_name=toy_index"],
    )
    inference_cfg = compose_stage_config(
        "inference",
        ["dataset=toy", "pipeline/inference@pipeline=dummy_keyword", "stage.run_name=toy_index"],
    )
    dev_mapping_cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "input_mapping=dev_tiny",
            "pipeline/inference@pipeline=dummy_keyword",
            "stage.run_name=toy_index",
        ],
    )
    evaluation_cfg = compose_stage_config("evaluation", ["dataset=toy"])

    assert indexing_cfg.dataset.name == "toy"
    assert "indexer" in indexing_cfg.pipeline.components
    assert inference_cfg.dataset.name == "toy"
    assert inference_cfg.input_mapping.type == "full_dataset"
    assert "retriever" in inference_cfg.pipeline.components
    assert dev_mapping_cfg.input_mapping.type == "generated"
    assert dev_mapping_cfg.input_mapping.name == "dev_tiny"
    assert evaluation_cfg.dataset.name == "toy"
    assert evaluation_cfg.dataset.qrels_path.endswith("data/processed/toy/qrels.jsonl")


def test_indexing_and_inference_run_names_are_optional_labels() -> None:
    indexing_cfg = compose_stage_config(
        "indexing",
        ["dataset=toy", "pipeline/indexing@pipeline=dummy_jsonl"],
    )
    inference_cfg = compose_stage_config(
        "inference",
        ["dataset=toy", "pipeline/inference@pipeline=dummy_keyword"],
    )

    assert indexing_cfg.stage.run_name is None
    assert inference_cfg.stage.run_name is None


def test_named_inference_run_updates_derived_paths_and_prediction_artifact() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "pipeline/inference@pipeline=dummy_keyword",
            "stage.run_name=bge",
        ],
    )

    original_run_id = str(cfg.stage.run_id)
    prepare_stage_run_config(cfg)

    assert cfg.stage.run_id == f"bge_{original_run_id}"
    assert str(cfg.stage.output_dir).endswith(f"artifacts/runs/inference/bge_{original_run_id}")
    assert str(cfg.stage.predictions_path).endswith(
        f"artifacts/runs/inference/bge_{original_run_id}/predictions.json"
    )
