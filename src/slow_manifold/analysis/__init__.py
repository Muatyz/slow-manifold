"""Structured behavioral and dynamical analyses."""

from .checkpoint_selection import (
    CheckpointSelectionError,
    CheckpointSelectionResult,
    RepresentativeSelectionConfig,
    SelectedCheckpoint,
    select_representative_checkpoints,
)
from .latent_dynamics import (
    AnalysisConfig,
    LatentDynamicsResult,
    NeighborhoodSamplingConfig,
    analyze_checkpoints,
)
from .speed_minima import (
    SpeedMinimumClassificationConfig,
    classify_speed_minimum,
)

__all__ = [
    "AnalysisConfig",
    "CheckpointSelectionError",
    "CheckpointSelectionResult",
    "LatentDynamicsResult",
    "NeighborhoodSamplingConfig",
    "RepresentativeSelectionConfig",
    "SelectedCheckpoint",
    "SpeedMinimumClassificationConfig",
    "analyze_checkpoints",
    "classify_speed_minimum",
    "select_representative_checkpoints",
]
