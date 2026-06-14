"""Console output helpers for CLI runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig

from retrieval_research.io import config_to_yaml


def print_stage_start(
    stage_name: str,
    cfg: DictConfig,
    *,
    overrides: Sequence[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Print a concise stage banner followed by the resolved config."""

    print(_rule())
    print(f"Stage: {stage_name}")
    print(f"Dry run: {dry_run}")
    if overrides:
        print("Overrides:")
        for override in overrides:
            print(f"  - {override}")

    for line in _summary_lines(cfg):
        print(line)

    print()
    print("Resolved config:")
    print(config_to_yaml(cfg).rstrip())
    print(_rule())


def _summary_lines(cfg: DictConfig) -> list[str]:
    lines: list[str] = []

    dataset = cfg.get("dataset")
    if dataset:
        lines.append(f"Dataset: {dataset.get('name', '<unnamed>')}")
        for key in ("documents_path", "queries_path", "qrels_path", "input_mapping_path"):
            value = dataset.get(key)
            if value:
                lines.append(f"  {key}: {value}")

    stage = cfg.get("stage")
    if stage:
        for key in ("output_dir", "predictions_path", "metrics_path"):
            value = stage.get(key)
            if value:
                lines.append(f"{key}: {value}")

    pipeline = cfg.get("pipeline")
    if pipeline and pipeline.get("components"):
        component_names = ", ".join(str(name) for name in pipeline.components.keys())
        lines.append(f"Pipeline components: {component_names}")

    return lines


def _rule() -> str:
    return "=" * 88


def print_stage_result(stage_name: str, result: Any) -> None:
    print(_rule())
    print(f"Stage complete: {stage_name}")
    print(result)
    print(_rule())
