"""Shared data structures and time-grid validation for behavioral tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class TaskConfigError(ValueError):
    """Raised when task timing or sampling parameters are inconsistent."""


@dataclass(frozen=True)
class SquarePulseStimulusConfig:
    """Common S1/S2/Go square-pulse encoding used by timing tasks."""

    waveform: str
    encoding: str
    width: float
    amplitude: float

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        dt: float,
    ) -> "SquarePulseStimulusConfig":
        required = {"waveform", "encoding", "width", "amplitude"}
        missing = required - set(data)
        if missing:
            raise TaskConfigError(
                "Missing stimulus fields: " + ", ".join(sorted(missing))
            )
        config = cls(
            waveform=str(data["waveform"]),
            encoding=str(data["encoding"]),
            width=float(data["width"]),
            amplitude=float(data["amplitude"]),
        )
        config.validate(dt)
        return config

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("cue",)

    @property
    def input_size(self) -> int:
        return 1

    def validate(self, dt: float) -> None:
        if self.waveform != "square":
            raise TaskConfigError("Only the 'square' stimulus waveform is supported")
        if self.encoding != "shared":
            raise TaskConfigError("stimulus.encoding must be 'shared'")
        if self.amplitude <= 0:
            raise TaskConfigError("stimulus amplitude must be positive")
        to_steps(self.width, dt, "stimulus.width")

    def render(
        self,
        *,
        cue_steps: Sequence[int],
        trial_steps: int,
        dt: float,
    ) -> FloatArray:
        """Render S1, S2, and Go on one shared scalar input channel."""
        if len(cue_steps) != 3:
            raise ValueError("cue_steps must contain S1, S2, and Go")
        pulse_steps = to_steps(self.width, dt, "stimulus.width")
        inputs = np.zeros((trial_steps, self.input_size), dtype=np.float64)
        for start in cue_steps:
            stop = int(start) + pulse_steps
            if start < 0 or stop > trial_steps:
                raise ValueError("cue pulse lies outside the trial")
            inputs[int(start) : stop, 0] = self.amplitude
        return inputs


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
    input_names: tuple[str, ...]
    loss_reduction: str = "weighted_mean"


@dataclass(frozen=True)
class PhaseNormalizedLossConfig:
    """Train-owned weights for task-defined, duration-normalized phases."""

    name: str
    phase_weights: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "PhaseNormalizedLossConfig":
        if data is None:
            raise TaskConfigError("train.loss must be a mapping")
        if not isinstance(data, Mapping):
            raise TaskConfigError("train.loss must be a mapping")
        if data.get("name") != "phase_normalized_mse":
            raise TaskConfigError("Only phase_normalized_mse is supported")
        raw_weights = data.get("phase_weights")
        if not isinstance(raw_weights, list) or not raw_weights:
            raise TaskConfigError("train.loss.phase_weights must be a non-empty list")

        weights: list[tuple[str, float]] = []
        for item in raw_weights:
            if not isinstance(item, Mapping):
                raise TaskConfigError("Each phase weight must be a mapping")
            unknown = set(item) - {"phase", "lambda"}
            if unknown or "phase" not in item or "lambda" not in item:
                raise TaskConfigError(
                    "Each phase weight requires only 'phase' and 'lambda'"
                )
            phase = str(item["phase"])
            try:
                weight = float(item["lambda"])
            except (TypeError, ValueError) as error:
                raise TaskConfigError("Loss lambda must be numeric") from error
            if not phase or not np.isfinite(weight) or weight < 0:
                raise TaskConfigError(
                    "Loss phase names must be non-empty and lambda finite and >= 0"
                )
            weights.append((phase, weight))

        phases = [phase for phase, _ in weights]
        if len(phases) != len(set(phases)):
            raise TaskConfigError("Loss phase names must be unique")
        if not any(weight > 0 for _, weight in weights):
            raise TaskConfigError("At least one loss lambda must be positive")
        return cls(name="phase_normalized_mse", phase_weights=tuple(weights))

    @classmethod
    def unit_weights(cls, *phases: str) -> "PhaseNormalizedLossConfig":
        """Provide unit lambdas for task-only sanity checks without train config."""
        return cls(
            name="phase_normalized_mse",
            phase_weights=tuple((phase, 1.0) for phase in phases),
        )

    def require_phases(self, *expected: str) -> dict[str, float]:
        weights = dict(self.phase_weights)
        if set(weights) != set(expected):
            raise TaskConfigError(
                "Loss phases must be exactly: " + ", ".join(expected)
            )
        return weights


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
