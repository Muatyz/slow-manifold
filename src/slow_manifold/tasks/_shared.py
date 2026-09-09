"""Shared data structures and time-grid validation for behavioral tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class TaskConfigError(ValueError):
    """Raised when task timing or sampling parameters are inconsistent."""


@dataclass(frozen=True)
class TrialMetadata:
    s1_step: int
    s2_step: int
    go_step: int
    response_step: int
    trial_steps: int
    interval: float
    delay: float
    split: str
    class_label: int | None = None


@dataclass(frozen=True)
class Trial:
    inputs: FloatArray
    target: FloatArray
    loss_mask: FloatArray
    metadata: TrialMetadata


@dataclass(frozen=True)
class TaskBatch:
    inputs: FloatArray
    target: FloatArray
    loss_mask: FloatArray
    valid_mask: BoolArray
    metadata: tuple[TrialMetadata, ...]
    dt: float
    input_names: tuple[str, ...] = ("S1", "S2", "Go")
    loss_reduction: str = "weighted_mean"


def pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TaskConfigError("Expected a two-element range")
    return float(value[0]), float(value[1])


def to_steps(value: float, dt: float, name: str, *, allow_zero: bool = False) -> int:
    steps = int(round(float(value) / dt))
    minimum = 0 if allow_zero else 1
    if steps < minimum:
        raise TaskConfigError(f"{name} must contain at least {minimum} time steps")
    if not np.isclose(steps * dt, value, rtol=0.0, atol=1e-9):
        raise TaskConfigError(f"{name}={value} is not aligned to dt={dt}")
    return steps
