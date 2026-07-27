"""Explicit preparation stage for reusable generated inference mappings."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from retrieval_core.input_mapping import (
    metadata_path_for,
    prepare_generated_input_mapping,
)
from retrieval_core.stages.base import StageContext

logger = logging.getLogger(__name__)


def run_prepare_mapping(
    cfg: DictConfig,
    *,
    context: StageContext | None = None,
) -> dict:
    generated, mapping_path = prepare_generated_input_mapping(cfg)
    metadata_path = metadata_path_for(mapping_path)
    result = {
        "mapping_path": str(mapping_path),
        "metadata_path": str(metadata_path),
        "query_count": len(generated.mapping),
    }

    context = context or StageContext.create(cfg)
    context.write_result(result)
    context.write_manifest(
        artifacts={
            "input_mapping": mapping_path,
            "input_mapping_metadata": metadata_path,
        },
        inputs={"dataset": str(cfg.dataset.name)},
    )
    logger.info(
        "Input mapping prepared: dataset=%s queries=%d mapping_path=%s",
        cfg.dataset.name,
        len(generated.mapping),
        mapping_path,
    )
    return result
