"""Compatibility exports for document fusion components."""

from retrieval_research.components.fusion.reciprocal_rank_fusion import ReciprocalRankFusion
from retrieval_research.components.fusion.score_fusion import ScoreFusion

__all__ = [
    "ReciprocalRankFusion",
    "ScoreFusion",
]
