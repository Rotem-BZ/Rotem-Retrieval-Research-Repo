"""Console output helpers for CLI runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig

from retrieval_core.utils.io import config_to_yaml


def print_stage_start(
    stage_name: str,
    cfg: DictConfig,
    *,
    overrides: Sequence[str] | None = None,
) -> None:
    """Print a concise stage banner followed by the resolved config."""

    print("=" * 88)
    print(f"Stage: {stage_name}")
    if overrides:
        print("Overrides:")
        for override in overrides:
            print(f"  - {override}")

    dataset = cfg.get("dataset")
    if dataset:
        print(f"Dataset: {dataset.get('name', '<unnamed>')}")
        for key in ("documents_path", "queries_path", "qrels_path"):
            value = dataset.get(key)
            if value:
                print(f"  {key}: {value}")

    input_mapping_recipe = cfg.get("input_mapping_recipe")
    if input_mapping_recipe:
        print(f"Input mapping recipe: {input_mapping_recipe.get('name', '<unnamed>')}")

    selections = cfg.get("selections")
    if selections and selections.get("input_mapping"):
        print(f"Input mapping: {selections.input_mapping}")
    if selections and selections.get("index_id"):
        print(f"Index id: {selections.index_id}")

    stage = cfg.get("stage")
    if stage:
        for key in (
            "output_dir",
            "inference_run_id",
            "predictions_path",
            "metrics_path",
        ):
            value = stage.get(key)
            if value:
                print(f"{key}: {value}")

    pipeline = cfg.get("pipeline")
    if pipeline and pipeline.get("components"):
        component_names = ", ".join(str(name) for name in pipeline.components.keys())
        print(f"Pipeline components: {component_names}")

    print()
    print("Resolved config:")
    print(config_to_yaml(cfg).rstrip())
    print("=" * 88)


def print_stage_result(stage_name: str, result: Any) -> None:
    print("=" * 88)
    print(f"Stage complete: {stage_name}")
    print(result)
    print("=" * 88)
