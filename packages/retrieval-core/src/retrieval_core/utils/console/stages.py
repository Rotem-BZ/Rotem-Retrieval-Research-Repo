"""Compatibility helpers that report stage lifecycle events through logging."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def print_stage_start(
    stage_name: str,
    cfg: DictConfig,
    *,
    overrides: Sequence[str] | None = None,
) -> None:
    """Report a stage start without exposing raw configuration values."""

    del overrides
    dataset = cfg.get("dataset")
    logger.info(
        "Stage started: stage=%s run_id=%s dataset=%s output_dir=%s",
        stage_name,
        cfg.get("stage", {}).get("run_id"),
        dataset.get("name", "<none>") if dataset else "<none>",
        cfg.get("stage", {}).get("output_dir"),
    )


def print_stage_result(stage_name: str, result: Any) -> None:
    """Report a stage completion for legacy callers."""

    del result
    logger.info("Stage completed: stage=%s", stage_name)
