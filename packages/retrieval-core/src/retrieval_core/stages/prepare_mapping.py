"""Explicit preparation stage for reusable generated inference mappings."""

from __future__ import annotations

import logging

from retrieval_core.input_mapping import (
    metadata_path_for,
    prepare_generated_input_mapping,
)
from retrieval_core.stages.base import Stage

logger = logging.getLogger(__name__)


class PrepareMappingStage(Stage):
    """Materialize a reusable generated inference mapping."""

    def prepare_config(self) -> None:
        pass

    def run(self) -> dict:
        generated, mapping_path = prepare_generated_input_mapping(self.cfg)
        metadata_path = metadata_path_for(mapping_path)
        result = {
            "mapping_path": str(mapping_path),
            "metadata_path": str(metadata_path),
            "query_count": len(generated.mapping),
        }

        self.write_result(result)
        self.write_manifest(
            artifacts={
                "input_mapping": mapping_path,
                "input_mapping_metadata": metadata_path,
            },
            inputs={"dataset": str(self.cfg.dataset.name)},
        )
        logger.info(
            "Input mapping prepared: dataset=%s queries=%d mapping_path=%s",
            self.cfg.dataset.name,
            len(generated.mapping),
            mapping_path,
        )
        return result
