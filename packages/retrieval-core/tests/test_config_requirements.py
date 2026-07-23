import re

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from retrieval_core.stages.base import prepare_stage_run_config
from retrieval_core.utils.config import compose_stage_config


def test_indexing_requires_dataset_and_pipeline_selection() -> None:
    with pytest.raises(ConfigCompositionException, match="pipeline/indexing@pipeline"):
        compose_stage_config("indexing", ["runtime=gpu"])

    with pytest.raises(ConfigCompositionException, match="dataset"):
        compose_stage_config(
            "indexing",
            ["pipeline/indexing@pipeline=scaffold/documents_jsonl", "runtime=gpu"],
        )


def test_inference_requires_dataset_and_pipeline_selection() -> None:
    with pytest.raises(ConfigCompositionException, match="pipeline/inference@pipeline"):
        compose_stage_config("inference", ["dataset=toy", "runtime=gpu"])

    with pytest.raises(ConfigCompositionException, match="dataset"):
        compose_stage_config(
            "inference",
            ["pipeline/inference@pipeline=scaffold/keyword_jsonl", "runtime=gpu"],
        )


def test_indexing_and_inference_require_runtime_selection() -> None:
    with pytest.raises(ConfigCompositionException, match="runtime"):
        compose_stage_config(
            "indexing",
            ["dataset=toy", "pipeline/indexing@pipeline=scaffold/documents_jsonl"],
        )

    with pytest.raises(ConfigCompositionException, match="runtime"):
        compose_stage_config(
            "inference",
            ["dataset=toy", "pipeline/inference@pipeline=scaffold/keyword_jsonl"],
        )


def test_evaluation_requires_dataset_selection() -> None:
    with pytest.raises(ConfigCompositionException, match="dataset"):
        compose_stage_config("evaluation")


def test_prepare_mapping_requires_explicit_run_id() -> None:
    cfg = compose_stage_config(
        "prepare_mapping",
        ["dataset=toy", "input_mapping_recipe=dev_tiny"],
    )

    assert OmegaConf.is_missing(cfg.stage, "run_id")


def test_explicit_stage_selections_compose() -> None:
    indexing_cfg = compose_stage_config(
        "indexing",
        [
            "dataset=toy",
            "pipeline/indexing@pipeline=scaffold/documents_jsonl",
            "runtime=gpu",
            "selections.index_id=toy-index",
        ],
    )
    inference_cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "pipeline/inference@pipeline=scaffold/keyword_jsonl",
            "runtime=gpu",
        ],
    )
    mapping_cfg = compose_stage_config(
        "prepare_mapping",
        [
            "dataset=toy",
            "input_mapping_recipe=dev_tiny",
            "stage.run_id=toy_dev_tiny",
        ],
    )
    evaluation_cfg = compose_stage_config("evaluation", ["dataset=toy"])

    assert indexing_cfg.dataset.name == "toy"
    assert "indexer" in indexing_cfg.pipeline.components
    assert inference_cfg.dataset.name == "toy"
    assert "input_mapping" not in inference_cfg
    assert inference_cfg.selections.input_mapping is None
    assert "retriever" in inference_cfg.pipeline.components
    assert mapping_cfg.input_mapping_recipe.type == "generated"
    assert mapping_cfg.input_mapping_recipe.name == "dev_tiny"
    assert mapping_cfg.stage.run_id == "toy_dev_tiny"
    assert evaluation_cfg.dataset.name == "toy"
    assert evaluation_cfg.dataset.qrels_path.endswith("data/processed/toy/qrels.jsonl")


def test_runtime_profiles_select_gpu_or_cpu_device() -> None:
    common = ["dataset=toy", "pipeline/inference@pipeline=scaffold/keyword_jsonl"]

    gpu_cfg = compose_stage_config("inference", [*common, "runtime=gpu"])
    cpu_cfg = compose_stage_config("inference", [*common, "runtime=cpu"])

    assert gpu_cfg.runtime.device == {"type": "single", "device": "cuda"}
    assert cpu_cfg.runtime.device == {"type": "single", "device": "cpu"}
    assert gpu_cfg.runtime.concurrency_limit == cpu_cfg.runtime.concurrency_limit == 4
    assert gpu_cfg.runtime.query_concurrency_limit == 4


def test_inference_accepts_prepared_input_mapping_name() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=cpu",
            "selections.input_mapping=toy_dev_tiny",
            "pipeline/inference@pipeline=rerank/bi_encoder",
            "selections/embedding_model=e5/small_v2",
        ],
    )

    assert cfg.selections.input_mapping == "toy_dev_tiny"


def test_explicit_inference_run_id_updates_derived_paths_and_prediction_artifact() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "pipeline/inference@pipeline=scaffold/keyword_jsonl",
            "runtime=gpu",
            "stage.run_id=bge",
        ],
    )

    prepare_stage_run_config(cfg)

    assert cfg.stage.run_id == "bge"
    assert str(cfg.stage.output_dir).endswith("artifacts/runs/inference/bge")
    assert str(cfg.stage.predictions_path).endswith("artifacts/runs/inference/bge/predictions.json")


def test_generated_timestamp_run_id_can_be_reused_as_hydra_override() -> None:
    cfg = compose_stage_config("evaluation", ["dataset=toy"])
    run_id = str(cfg.stage.run_id)

    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}", run_id)

    overridden = compose_stage_config(
        "evaluation",
        ["dataset=toy", f"stage.run_id={run_id}"],
    )

    assert overridden.stage.run_id == run_id
