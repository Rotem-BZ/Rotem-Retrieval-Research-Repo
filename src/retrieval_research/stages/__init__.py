"""Stage runners for retrieval experiments."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from omegaconf import DictConfig

from retrieval_research.stages.evaluation import run_evaluation
from retrieval_research.stages.indexing import run_indexing
from retrieval_research.stages.inference import run_inference

StageResult = dict[str, Any] | list[dict[str, Any]]
StageRunner = Callable[[DictConfig], StageResult | Awaitable[StageResult]]

STAGE_RUNNERS: dict[str, StageRunner] = {
    "indexing": run_indexing,
    "inference": run_inference,
    "evaluation": run_evaluation,
}
