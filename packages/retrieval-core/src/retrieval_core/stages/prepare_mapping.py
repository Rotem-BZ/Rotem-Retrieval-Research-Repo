"""Explicit preparation stage for reusable generated inference mappings."""

from __future__ import annotations

from omegaconf import DictConfig

from retrieval_core.input_mapping import (
    metadata_path_for,
    prepare_generated_input_mapping,
)
from retrieval_core.stages.base import StageContext, is_dry_run


def run_prepare_mapping(cfg: DictConfig) -> dict:
    generated, mapping_path, reused = prepare_generated_input_mapping(
        cfg,
        persist=not is_dry_run(cfg),
    )
    metadata_path = metadata_path_for(mapping_path)
    result = {
        "mapping_path": str(mapping_path),
        "metadata_path": str(metadata_path),
        "query_count": len(generated.mapping),
        "reused": reused,
    }

    context = StageContext.from_config(cfg)
    context.write_resolved_config()
    context.write_result(result)
    context.write_manifest(
        artifacts={
            "input_mapping": mapping_path,
            "input_mapping_metadata": metadata_path,
        },
        inputs={"dataset": str(cfg.dataset.name)},
    )
    return result
