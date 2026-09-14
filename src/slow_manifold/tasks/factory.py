"""Construct configured behavioral tasks through one public dispatch path."""

from __future__ import annotations

from typing import Any, Mapping

from ._shared import PhaseNormalizedLossConfig, TaskConfigError
from .interval_categorization import (
    IntervalCategorizationConfig,
    IntervalCategorizationTask,
)
from .interval_reproduction import IntervalReproductionConfig, IntervalReproductionTask

ConfiguredTask = IntervalCategorizationTask | IntervalReproductionTask


def create_task(
    data: Mapping[str, Any],
    *,
    dt: float,
    split: str = "train",
    loss: PhaseNormalizedLossConfig | None = None,
) -> ConfiguredTask:
    """Build the task selected by its resolved component ``name``."""
    name = data.get("name")
    if name == "delayed_interval_categorization":
        config = IntervalCategorizationConfig.from_mapping(data, dt=dt)
        return IntervalCategorizationTask(config, split=split, loss=loss)
    if name == "delayed_interval_reproduction":
        config = IntervalReproductionConfig.from_mapping(data, dt=dt)
        return IntervalReproductionTask(config, split=split, loss=loss)
    raise TaskConfigError(f"Unknown task name: {name!r}")
