"""Post-training selection of representative checkpoints.

The selector discovers presentation milestones from the completed validation
trace.  It does not claim that a checkpoint is an optimum or that a detected
transition has a particular dynamical mechanism.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from slow_manifold.config import dump_yaml


class CheckpointSelectionError(ValueError):
    """Raised when representative-checkpoint selection cannot be resolved."""


@dataclass(frozen=True)
class RepresentativeSelectionConfig:
    """Rules for automatic selection, or an explicit manual replacement."""

    mode: str = "auto"
    metric: str = "validation_loss"
    direction: str = "minimize"
    transform: str = "log"
    max_epochs: int = 5
    smoothing_points: int = 5
    transition_radius_points: int = 3
    baseline_points: int = 20
    abrupt_effect_mad: float = 5.0
    abrupt_slope_ratio: float = 5.0
    plateau_points: int = 50
    plateau_max_relative_improvement: float = 0.02
    progress_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    manual_epochs: tuple[int, ...] = ()

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "RepresentativeSelectionConfig":
        values = {} if data is None else data
        if not isinstance(values, Mapping):
            raise CheckpointSelectionError(
                "representative_selection must be a mapping"
            )
        try:
            config = cls(
                mode=str(values.get("mode", "auto")),
                metric=str(values.get("metric", "validation_loss")),
                direction=str(values.get("direction", "minimize")),
                transform=str(values.get("transform", "log")),
                max_epochs=int(values.get("max_epochs", 5)),
                smoothing_points=int(values.get("smoothing_points", 5)),
                transition_radius_points=int(
                    values.get("transition_radius_points", 3)
                ),
                baseline_points=int(values.get("baseline_points", 20)),
                abrupt_effect_mad=float(values.get("abrupt_effect_mad", 5.0)),
                abrupt_slope_ratio=float(
                    values.get("abrupt_slope_ratio", 5.0)
                ),
                plateau_points=int(values.get("plateau_points", 50)),
                plateau_max_relative_improvement=float(
                    values.get("plateau_max_relative_improvement", 0.02)
                ),
                progress_fractions=tuple(
                    float(value)
                    for value in values.get(
                        "progress_fractions", (0.25, 0.5, 0.75)
                    )
                ),
                manual_epochs=tuple(
                    int(epoch) for epoch in values.get("manual_epochs", ())
                ),
            )
        except (TypeError, ValueError) as error:
            raise CheckpointSelectionError(
                f"Invalid representative selection configuration: {error}"
            ) from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in {"auto", "manual"}:
            raise CheckpointSelectionError("selection mode must be auto or manual")
        if self.direction not in {"minimize", "maximize"}:
            raise CheckpointSelectionError(
                "selection direction must be minimize or maximize"
            )
        if self.transform not in {"identity", "log"}:
            raise CheckpointSelectionError(
                "selection transform must be identity or log"
            )
        if self.max_epochs < 2:
            raise CheckpointSelectionError("max_epochs must be at least 2")
        if self.smoothing_points < 1 or self.smoothing_points % 2 == 0:
            raise CheckpointSelectionError(
                "smoothing_points must be a positive odd integer"
            )
        if self.transition_radius_points < 1 or self.baseline_points < 2:
            raise CheckpointSelectionError(
                "transition_radius_points and baseline_points are too small"
            )
        if self.abrupt_effect_mad <= 0 or self.abrupt_slope_ratio <= 0:
            raise CheckpointSelectionError("abrupt thresholds must be positive")
        if self.plateau_points < 4:
            raise CheckpointSelectionError("plateau_points must be at least 4")
        if not 0 <= self.plateau_max_relative_improvement < 1:
            raise CheckpointSelectionError(
                "plateau_max_relative_improvement must lie in [0, 1)"
            )
        if any(not 0 < value < 1 for value in self.progress_fractions):
            raise CheckpointSelectionError(
                "progress_fractions must lie strictly between 0 and 1"
            )
        if tuple(sorted(set(self.manual_epochs))) != self.manual_epochs:
            raise CheckpointSelectionError(
                "manual_epochs must be unique and strictly increasing"
            )
        if any(epoch < 0 for epoch in self.manual_epochs):
            raise CheckpointSelectionError("manual_epochs must be non-negative")
        if self.mode == "manual" and not self.manual_epochs:
            raise CheckpointSelectionError(
                "manual mode requires at least one manual epoch"
            )

    def with_manual_epochs(
        self, epochs: Sequence[int]
    ) -> "RepresentativeSelectionConfig":
        """Return an explicitly manual selection using the same base rules."""
        config = replace(
            self,
            mode="manual",
            manual_epochs=tuple(int(epoch) for epoch in epochs),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class SelectedCheckpoint:
    label: str
    role: str
    epoch: int
    metric_value: float
    smoothed_metric_value: float
    anchor_epoch: int
    anchor_smoothed_metric_value: float
    selection_method: str
    window_epoch_min: int
    window_epoch_max: int
    reason: str


@dataclass(frozen=True)
class CheckpointSelectionResult:
    metadata_path: Path
    selected: tuple[SelectedCheckpoint, ...]
    transition_classification: str
    terminal_regime: str

    @property
    def epochs(self) -> tuple[int, ...]:
        return tuple(item.epoch for item in self.selected)

    @property
    def labels(self) -> dict[int, str]:
        return {
            item.epoch: f"{item.label} {item.role.replace('_', ' ')}"
            for item in self.selected
        }


def select_representative_checkpoints(
    *,
    metrics_path: str | Path,
    checkpoint_epochs: Sequence[int],
    config: RepresentativeSelectionConfig,
    output_path: str | Path,
    source: str,
) -> CheckpointSelectionResult:
    """Select available checkpoints and write a traceable selection manifest."""
    metric_path = Path(metrics_path)
    epochs, values = _read_metric(metric_path, config.metric)
    by_epoch = {int(epoch): float(value) for epoch, value in zip(epochs, values)}
    available = tuple(sorted(set(int(epoch) for epoch in checkpoint_epochs)))
    missing_metrics = [epoch for epoch in available if epoch not in by_epoch]
    if missing_metrics:
        raise CheckpointSelectionError(
            "Checkpoint epochs missing from metrics.csv: "
            + ", ".join(str(epoch) for epoch in missing_metrics[:5])
        )
    candidate_epochs = np.asarray(available, dtype=np.int64)
    candidate_values = np.asarray(
        [by_epoch[epoch] for epoch in available], dtype=np.float64
    )
    if candidate_epochs.size < 2:
        raise CheckpointSelectionError(
            "At least two checkpoint epochs are required for selection"
        )

    objective = _objective_values(candidate_values, config)
    smoothed = _rolling_median(objective, config.smoothing_points)
    displayed_smoothed = _inverse_objective(smoothed, config)

    if config.mode == "manual":
        selected_indices = _manual_indices(candidate_epochs, config.manual_epochs)
        roles = [
            "initialization" if candidate_epochs[index] == 0 else "manual"
            for index in selected_indices
        ]
        reasons = ["explicit manual epoch" for _ in selected_indices]
        choices = [
            _RepresentativeChoice(
                index=index,
                anchor_index=index,
                window_start=index,
                window_end=index,
                method="exact_manual_epoch",
            )
            for index in selected_indices
        ]
        transition = {
            "classification": "not_evaluated_manual_selection",
            "reason": "Manual selection bypasses automatic transition labeling.",
        }
    else:
        selected_indices, roles, reasons, transition = _automatic_indices(
            candidate_epochs, smoothed, objective, config
        )
        choices = _resolve_automatic_representatives(
            selected_indices,
            roles,
            objective=objective,
            smoothed=smoothed,
            smoothing_points=config.smoothing_points,
        )

    terminal = _terminal_regime(candidate_values, config)
    selected = tuple(
        SelectedCheckpoint(
            label=f"({_roman(position + 1)})",
            role=role,
            epoch=int(candidate_epochs[choice.index]),
            metric_value=float(candidate_values[choice.index]),
            smoothed_metric_value=float(displayed_smoothed[choice.index]),
            anchor_epoch=int(candidate_epochs[choice.anchor_index]),
            anchor_smoothed_metric_value=float(
                displayed_smoothed[choice.anchor_index]
            ),
            selection_method=choice.method,
            window_epoch_min=int(candidate_epochs[choice.window_start]),
            window_epoch_max=int(candidate_epochs[choice.window_end]),
            reason=reason,
        )
        for position, (choice, role, reason) in enumerate(
            zip(choices, roles, reasons)
        )
    )
    destination = Path(output_path)
    dump_yaml(
        {
            "schema_version": 2,
            "source": source,
            "selection_mode": config.mode,
            "metric": config.metric,
            "direction": config.direction,
            "transform": config.transform,
            "metrics_path": str(metric_path.resolve()),
            "metrics_sha256": _sha256(metric_path),
            "candidate_checkpoint_count": len(available),
            "candidate_epoch_min": int(candidate_epochs[0]),
            "candidate_epoch_max": int(candidate_epochs[-1]),
            "median_checkpoint_spacing": float(
                np.median(np.diff(candidate_epochs))
            ),
            "config": asdict(config),
            "transition": transition,
            "terminal_regime": terminal,
            "selected": [asdict(item) for item in selected],
            "interpretation": {
                "roman_labels": "display order only; roles carry the semantics",
                "abrupt": (
                    "descriptive metric change supported by configured effect and "
                    "slope-ratio thresholds; not evidence of a dynamical mechanism"
                ),
                "quasi_plateau": (
                    "small observed terminal improvement; not proof of an optimum"
                ),
                "selection": (
                    "the smoothed trace locates each stage; the displayed epoch is "
                    "an actual checkpoint selected inside its recorded local window"
                ),
                "mature": (
                    "best raw checkpoint inside the best smoothed post-transition "
                    "neighborhood observed in this run"
                ),
            },
        },
        destination,
    )
    return CheckpointSelectionResult(
        metadata_path=destination,
        selected=selected,
        transition_classification=str(transition["classification"]),
        terminal_regime=str(terminal["classification"]),
    )


@dataclass(frozen=True)
class _RepresentativeChoice:
    index: int
    anchor_index: int
    window_start: int
    window_end: int
    method: str


def _read_metric(path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise CheckpointSelectionError(
                f"Metric {metric!r} is not present in {path}"
            )
        rows = list(reader)
    samples: list[tuple[int, float]] = []
    for row in rows:
        raw_value = row.get(metric)
        if raw_value is None or not raw_value.strip():
            continue
        try:
            samples.append((int(row["epoch"]), float(raw_value)))
        except (TypeError, ValueError) as error:
            raise CheckpointSelectionError(
                f"Metric {metric!r} contains an invalid value"
            ) from error
    epochs = np.asarray([epoch for epoch, _ in samples], dtype=np.int64)
    values = np.asarray([value for _, value in samples], dtype=np.float64)
    if epochs.size == 0 or np.any(np.diff(epochs) <= 0):
        raise CheckpointSelectionError("Metric epochs must be nonempty and increasing")
    if not np.isfinite(values).all():
        raise CheckpointSelectionError(
            f"Metric {metric!r} contains non-finite values"
        )
    return epochs, values


def _objective_values(
    values: np.ndarray, config: RepresentativeSelectionConfig
) -> np.ndarray:
    transformed = values.copy()
    if config.transform == "log":
        if np.any(transformed <= 0):
            raise CheckpointSelectionError(
                f"Metric {config.metric!r} must be positive for log transform"
            )
        transformed = np.log(transformed)
    return transformed if config.direction == "minimize" else -transformed


def _inverse_objective(
    values: np.ndarray, config: RepresentativeSelectionConfig
) -> np.ndarray:
    transformed = values if config.direction == "minimize" else -values
    return np.exp(transformed) if config.transform == "log" else transformed


def _rolling_median(values: np.ndarray, requested_points: int) -> np.ndarray:
    radius = min(requested_points // 2, max(0, (len(values) - 1) // 2))
    if radius == 0:
        return values.copy()
    return np.asarray(
        [
            np.median(values[max(0, index - radius) : index + radius + 1])
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def _manual_indices(
    candidate_epochs: np.ndarray, requested_epochs: Sequence[int]
) -> list[int]:
    epoch_to_index = {
        int(epoch): index for index, epoch in enumerate(candidate_epochs)
    }
    missing = [int(epoch) for epoch in requested_epochs if epoch not in epoch_to_index]
    if missing:
        raise CheckpointSelectionError(
            "Manual representative epochs have no stored checkpoint: "
            + ", ".join(str(epoch) for epoch in missing)
        )
    return [epoch_to_index[int(epoch)] for epoch in requested_epochs]


def _resolve_automatic_representatives(
    anchor_indices: Sequence[int],
    roles: Sequence[str],
    *,
    objective: np.ndarray,
    smoothed: np.ndarray,
    smoothing_points: int,
) -> list[_RepresentativeChoice]:
    """Resolve trend milestones to robust, actually displayed checkpoints.

    Phase boundaries stay fixed because moving them would change their temporal
    meaning. Other milestones use a non-overlapping neighborhood around their
    smoothed-trace anchor. The mature panel uses the best raw metric in that
    stable neighborhood; intermediate panels use the raw checkpoint closest to
    the local smoothed level so that a transient spike is not displayed as the
    phase representative.
    """
    if len(anchor_indices) != len(roles):
        raise CheckpointSelectionError(
            "Representative anchors and roles must have equal length"
        )
    if not anchor_indices:
        return []

    radius = smoothing_points // 2
    exact_roles = {"initialization", "pre_transition", "post_transition"}
    choices: list[_RepresentativeChoice] = []
    for position, (anchor, role) in enumerate(zip(anchor_indices, roles)):
        anchor = int(anchor)
        if role in exact_roles or radius == 0:
            choices.append(
                _RepresentativeChoice(
                    index=anchor,
                    anchor_index=anchor,
                    window_start=anchor,
                    window_end=anchor,
                    method="phase_anchor",
                )
            )
            continue

        previous_anchor = (
            int(anchor_indices[position - 1]) if position > 0 else None
        )
        next_anchor = (
            int(anchor_indices[position + 1])
            if position + 1 < len(anchor_indices)
            else None
        )
        lower_boundary = (
            0
            if previous_anchor is None
            else (previous_anchor + anchor) // 2 + 1
        )
        upper_boundary = (
            len(objective) - 1
            if next_anchor is None
            else (anchor + next_anchor) // 2
        )
        window_start = max(lower_boundary, anchor - radius)
        window_end = min(upper_boundary, anchor + radius)
        candidate_indices = np.arange(window_start, window_end + 1)

        if role == "mature":
            local_values = objective[candidate_indices]
            best = float(np.min(local_values))
            matches = candidate_indices[
                np.isclose(local_values, best, rtol=1e-10, atol=1e-12)
            ]
            selected_index = int(matches[-1])
            method = "best_raw_in_best_smoothed_window"
        else:
            distances = np.abs(
                objective[candidate_indices] - float(smoothed[anchor])
            )
            best_distance = float(np.min(distances))
            matches = candidate_indices[
                np.isclose(distances, best_distance, rtol=1e-10, atol=1e-12)
            ]
            selected_index = min(
                (int(index) for index in matches),
                key=lambda index: (abs(index - anchor), index),
            )
            method = "closest_raw_to_smoothed_anchor"

        choices.append(
            _RepresentativeChoice(
                index=selected_index,
                anchor_index=anchor,
                window_start=window_start,
                window_end=window_end,
                method=method,
            )
        )
    return choices


def _automatic_indices(
    epochs: np.ndarray,
    smoothed: np.ndarray,
    objective: np.ndarray,
    config: RepresentativeSelectionConfig,
) -> tuple[list[int], list[str], list[str], dict[str, Any]]:
    event = _dominant_transition(epochs, smoothed, objective, config)
    if event["classification"] == "abrupt":
        center = int(event["candidate_index"])
        radius = int(event["radius_points"])
        pre = max(0, center - radius)
        post = min(len(epochs) - 1, center + radius)
        mature = _mature_index(smoothed, start=post)
        refinement = _fractional_progress_index(
            smoothed, start=post, end=mature, fraction=0.5
        )
        proposed = [(0, "initialization", "untrained network")]
        # When the transition occurs near initialization, the initial panel is
        # already an adequate pre-transition reference. Use the freed panel to
        # show the slowest intervening regime. For a late event, bracket it.
        early_cutoff = float(epochs[0]) + 0.05 * float(epochs[-1] - epochs[0])
        if float(epochs[pre]) <= early_cutoff:
            slow_index, slow_role, slow_reason = _slow_regime_index(
                epochs,
                smoothed,
                start=post,
                end=refinement,
                config=config,
            )
            proposed.extend(
                [
                    (
                        post,
                        "post_transition",
                        "first sampled state after the dominant change window",
                    ),
                    (slow_index, slow_role, slow_reason),
                ]
            )
        else:
            proposed.extend(
                [
                    (
                        pre,
                        "pre_transition",
                        "last sampled state before the dominant change window",
                    ),
                    (
                        post,
                        "post_transition",
                        "first sampled state after the dominant change window",
                    ),
                ]
            )
        proposed.extend(
            [
                (
                    refinement,
                    "mid_refinement",
                    "half of the observed post-transition improvement",
                ),
                (
                    mature,
                    "mature",
                    "representative of the best smoothed post-transition region",
                ),
            ]
        )
    else:
        proposed = [(0, "initialization", "untrained network")]
        for fraction, index in _progress_indices(smoothed, config.progress_fractions):
            proposed.append(
                (
                    index,
                    f"progress_{int(round(100 * fraction))}",
                    f"first checkpoint reaching {fraction:.0%} of observed progress",
                )
            )
        proposed.append(
            (
                _mature_index(smoothed),
                "mature",
                "representative of the best smoothed region in this run",
            )
        )

    unique: dict[int, tuple[str, str]] = {}
    for index, role, reason in proposed:
        unique.setdefault(int(index), (role, reason))
    ordered = sorted(unique.items())
    if len(ordered) > config.max_epochs:
        keep = np.linspace(0, len(ordered) - 1, config.max_epochs).round().astype(int)
        ordered = [ordered[index] for index in sorted(set(keep.tolist()))]
    indices = [index for index, _ in ordered]
    roles = [description[0] for _, description in ordered]
    reasons = [description[1] for _, description in ordered]
    return indices, roles, reasons, event


def _dominant_transition(
    epochs: np.ndarray,
    smoothed: np.ndarray,
    objective: np.ndarray,
    config: RepresentativeSelectionConfig,
) -> dict[str, Any]:
    n_points = len(epochs)
    radius = min(config.transition_radius_points, (n_points - 1) // 2)
    if radius < 1 or n_points < 2 * radius + 1:
        return {
            "classification": "insufficient_data",
            "reason": "Too few checkpoint samples to estimate a transition.",
        }

    # Detect acquisition on the best-so-far envelope. This prevents a transient
    # loss spike followed by mere recovery from being mislabeled as learning.
    trend = np.minimum.accumulate(smoothed)
    residual = objective - smoothed
    residual_mad = 1.4826 * float(
        np.median(np.abs(residual - np.median(residual)))
    )
    epsilon = np.finfo(np.float64).eps
    candidates: list[dict[str, float | int]] = []
    for center in range(radius, n_points - radius):
        left = center - radius
        right = center + radius
        pre_level = float(np.median(trend[left:center]))
        post_level = float(np.median(trend[center + 1 : right + 1]))
        improvement = pre_level - post_level
        if improvement <= 0:
            continue
        duration = max(float(epochs[right] - epochs[left]), 1.0)
        event_rate = improvement / duration
        baseline_start = max(0, left - config.baseline_points)
        baseline_slice = slice(baseline_start, max(left + 1, center))
        base_values = trend[baseline_slice]
        base_epochs = epochs[baseline_slice]
        if len(base_values) >= 2:
            baseline_rate = float(
                np.median(
                    np.abs(np.diff(base_values) / np.maximum(np.diff(base_epochs), 1))
                )
            )
        else:
            baseline_rate = 0.0
        effect_mad = float(improvement / max(residual_mad, epsilon))
        noise_rate = residual_mad / duration
        slope_ratio = float(
            event_rate / max(baseline_rate, noise_rate, epsilon)
        )
        score = float(effect_mad * np.sqrt(slope_ratio))
        candidates.append(
            {
                "candidate_index": center,
                "epoch": int(epochs[center]),
                "radius_points": radius,
                "improvement_transformed": float(improvement),
                "effect_mad": effect_mad,
                "slope_ratio": slope_ratio,
                "score": score,
            }
        )
    if not candidates:
        return {
            "classification": "gradual",
            "reason": "No improving change window was detected.",
        }
    best = max(candidates, key=lambda item: float(item["score"]))
    abrupt = (
        float(best["effect_mad"]) >= config.abrupt_effect_mad
        and float(best["slope_ratio"]) >= config.abrupt_slope_ratio
    )
    return {
        **best,
        "classification": "abrupt" if abrupt else "gradual",
        "thresholds": {
            "effect_mad": config.abrupt_effect_mad,
            "slope_ratio": config.abrupt_slope_ratio,
        },
        "reason": (
            "Both robust effect-size and slope-ratio thresholds were met."
            if abrupt
            else "The dominant change did not meet both abruptness thresholds."
        ),
    }


def _progress_indices(
    smoothed: np.ndarray, fractions: Sequence[float]
) -> list[tuple[float, int]]:
    running_best = np.minimum.accumulate(smoothed)
    initial = float(running_best[0])
    best = float(running_best.min())
    if initial <= best + np.finfo(np.float64).eps:
        return []
    result: list[tuple[float, int]] = []
    for fraction in fractions:
        target = initial - float(fraction) * (initial - best)
        matches = np.flatnonzero(running_best <= target)
        if matches.size:
            result.append((float(fraction), int(matches[0])))
    return result


def _mature_index(smoothed: np.ndarray, start: int = 0) -> int:
    first = min(max(int(start), 0), len(smoothed) - 1)
    tail = smoothed[first:]
    minimum = float(np.min(tail))
    # On a tied terminal plateau, prefer its latest sampled state so that the
    # mature panel is distinct from the immediate post-transition panel.
    matches = np.flatnonzero(np.isclose(tail, minimum, rtol=1e-10, atol=1e-12))
    return first + int(matches[-1])


def _fractional_progress_index(
    smoothed: np.ndarray, *, start: int, end: int, fraction: float
) -> int:
    if end <= start:
        return int(start)
    running_best = np.minimum.accumulate(smoothed[start : end + 1])
    initial = float(running_best[0])
    final = float(running_best[-1])
    if initial <= final + np.finfo(np.float64).eps:
        return int(start)
    target = initial - fraction * (initial - final)
    matches = np.flatnonzero(running_best <= target)
    return int(start + (matches[0] if matches.size else end - start))


def _slow_regime_index(
    epochs: np.ndarray,
    smoothed: np.ndarray,
    *,
    start: int,
    end: int,
    config: RepresentativeSelectionConfig,
) -> tuple[int, str, str]:
    if end <= start + 2:
        return (
            int(start),
            "post_transition",
            "insufficient separation to resolve an intermediate regime",
        )
    trend = np.minimum.accumulate(smoothed)
    span = end - start
    radius = min(max(1, config.plateau_points // 2), max(1, span // 3))
    candidates: list[tuple[float, float, int]] = []
    for center in range(start + radius, end - radius + 1):
        left = center - radius
        right = center + radius
        improvement = max(float(trend[left] - trend[right]), 0.0)
        duration = max(float(epochs[right] - epochs[left]), 1.0)
        if config.transform == "log":
            relative_improvement = 1.0 - float(np.exp(-improvement))
        else:
            relative_improvement = improvement / max(
                abs(float(trend[left])), np.finfo(np.float64).eps
            )
        candidates.append((relative_improvement / duration, relative_improvement, center))
    if not candidates:
        center = start + span // 2
        return (
            int(center),
            "intermediate_regime",
            "midpoint between post-transition and refinement milestones",
        )
    _, relative_improvement, center = min(candidates)
    if relative_improvement <= config.plateau_max_relative_improvement:
        return (
            int(center),
            "quasi_plateau",
            "lowest local best-so-far slope met the configured plateau tolerance",
        )
    return (
        int(center),
        "slow_learning_regime",
        "lowest local best-so-far slope remained above the plateau tolerance",
    )


def _terminal_regime(
    values: np.ndarray, config: RepresentativeSelectionConfig
) -> dict[str, Any]:
    points = min(config.plateau_points, len(values))
    if points < 4:
        return {
            "classification": "insufficient_data",
            "reason": "Too few checkpoint samples to assess the terminal regime.",
        }
    tail = values[-points:]
    quarter = max(1, points // 4)
    start = float(np.median(tail[:quarter]))
    end = float(np.median(tail[-quarter:]))
    scale = max(abs(start), np.finfo(np.float64).eps)
    improvement = (
        (start - end) / scale
        if config.direction == "minimize"
        else (end - start) / scale
    )
    threshold = config.plateau_max_relative_improvement
    if improvement < -threshold:
        classification = "regressing"
    elif improvement <= threshold:
        classification = "quasi_plateau"
    else:
        classification = "still_improving"
    return {
        "classification": classification,
        "points": points,
        "start_median": start,
        "end_median": end,
        "relative_improvement": improvement,
        "max_relative_improvement_for_plateau": threshold,
        "reason": "Terminal regime is descriptive and does not establish an optimum.",
    }


def _roman(number: int) -> str:
    numerals = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix")
    if not 1 <= number <= len(numerals):
        return str(number)
    return numerals[number - 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
