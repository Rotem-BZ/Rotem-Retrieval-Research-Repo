"""Stage classes for retrieval experiments."""

from retrieval_core.stages.base import Stage, StageResult
from retrieval_core.stages.evaluation import EvaluationStage
from retrieval_core.stages.indexing import IndexingStage
from retrieval_core.stages.inference import InferenceStage
from retrieval_core.stages.prepare_mapping import PrepareMappingStage

STAGES: dict[str, type[Stage]] = {
    "indexing": IndexingStage,
    "inference": InferenceStage,
    "evaluation": EvaluationStage,
    "prepare_mapping": PrepareMappingStage,
}

__all__ = ["STAGES", "Stage", "StageResult"]
