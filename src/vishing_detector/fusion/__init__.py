"""Multi-modal threat fusion and scoring."""

from vishing_detector.fusion.scorer import (
    CallExplanation,
    ChunkRiskAssessment,
    RiskScorer,
)

__all__ = ["CallExplanation", "ChunkRiskAssessment", "RiskScorer"]
