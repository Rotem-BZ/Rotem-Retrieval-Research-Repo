"""Stage runners for retrieval experiments."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from omegaconf import DictConfig

from retrieval_core.stages.evaluation import run_evaluation
from retrieval_core.stages.indexing import run_indexing
from retrieval_core.stages.inference import run_inference
from retrieval_core.stages.prepare_mapping import run_prepare_mapping

StageResult = dict[str, Any] | list[dict[str, Any]]
StageRunner = Callable[[DictConfig], StageResult | Awaitable[StageResult]]

STAGE_RUNNERS: dict[str, StageRunner] = {
    "indexing": run_indexing,
    "inference": run_inference,
    "evaluation": run_evaluation,
    "prepare_mapping": run_prepare_mapping,
}
