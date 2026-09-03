"""Coordinate-aligned rank-2 latent vector fields across checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from slow_manifold.config import dump_yaml
from slow_manifold.models import Rank2CTRNN, Rank2CTRNNConfig
from slow_manifold.tasks import IntervalCategorizationTask, TaskBatch
from slow_manifold.training.checkpoint import load_checkpoint


class AnalysisConfigError(ValueError):
    """Raised when training-dynamics analysis configuration is invalid."""


@dataclass(frozen=True)
class EvaluationTrialConfig:
    interval: float
    delay: float
    s1_onset: float


@dataclass(frozen=True)
class AnalysisConfig:
    name: str
    input_condition: tuple[float, ...]
    representative_epochs: tuple[int, ...]
    grid_points: int
    coordinate_bounds: tuple[float, float, float, float] | None
    padding_fraction: float
    bounds_expansion_fraction: float
    max_bounds_expansions: int
    speed_floor: float
    evaluation_trials: tuple[EvaluationTrialConfig, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AnalysisConfig":
        try:
            raw_bounds = data["coordinate_bounds"]
            bounds = (
                None
                if raw_bounds is None
                else tuple(float(value) for value in raw_bounds)
            )
            raw_trials = data["evaluation_trials"]
            trials = tuple(
                EvaluationTrialConfig(
                    interval=float(item["interval"]),
                    delay=float(item["delay"]),
                    s1_onset=float(item["s1_onset"]),
                )
                for item in raw_trials
            )
            config = cls(
                name=str(data["name"]),
                input_condition=tuple(float(value) for value in data["input_condition"]),
                representative_epochs=tuple(
                    int(epoch) for epoch in data["representative_epochs"]
                ),
                grid_points=int(data["grid_points"]),
                coordinate_bounds=bounds,  # type: ignore[arg-type]
                padding_fraction=float(data["padding_fraction"]),
                bounds_expansion_fraction=float(
                    data.get("bounds_expansion_fraction", 0.5)
                ),
                max_bounds_expansions=int(data.get("max_bounds_expansions", 2)),
                speed_floor=float(data["speed_floor"]),
                evaluation_trials=trials,
            )
        except (KeyError, TypeError) as error:
            raise AnalysisConfigError(f"Invalid analysis configuration: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.grid_points < 5:
            raise AnalysisConfigError("grid_points must be at least 5")
        if self.padding_fraction < 0 or self.speed_floor <= 0:
            raise AnalysisConfigError("Invalid padding_fraction or speed_floor")
        if self.bounds_expansion_fraction <= 0 or self.max_bounds_expansions < 0:
            raise AnalysisConfigError(
                "bounds_expansion_fraction must be positive and "
                "max_bounds_expansions non-negative"
            )
        if not self.evaluation_trials:
            raise AnalysisConfigError("evaluation_trials must not be empty")
        if any(epoch < 0 for epoch in self.representative_epochs):
            raise AnalysisConfigError("representative_epochs must be non-negative")
        if len(set(self.representative_epochs)) != len(self.representative_epochs):
            raise AnalysisConfigError("representative_epochs must be unique")
        if self.coordinate_bounds is not None:
            if len(self.coordinate_bounds) != 4:
                raise AnalysisConfigError("coordinate_bounds must have four values")
            x_min, x_max, y_min, y_max = self.coordinate_bounds
            if x_min >= x_max or y_min >= y_max:
                raise AnalysisConfigError("coordinate_bounds must be increasing")


@dataclass(frozen=True)
class LatentDynamicsResult:
    data_path: Path
    metadata_path: Path
    epochs: NDArray[np.int64]
    representative_epochs: tuple[int, ...]


def analyze_checkpoints(
    *,
    checkpoint_paths: Mapping[int, Path],
    model_config: Rank2CTRNNConfig,
    task: IntervalCategorizationTask,
    config: AnalysisConfig,
    output_dir: Path,
) -> LatentDynamicsResult:
    """Evaluate checkpoints in a sequentially aligned exact latent basis."""
    if len(config.input_condition) != model_config.input_size:
        raise AnalysisConfigError("input_condition length must match model input_size")
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_batch = _evaluation_batch(task, config.evaluation_trials)
    inputs = torch.as_tensor(evaluation_batch.inputs, dtype=model_config.torch_dtype)
    epochs = np.array(sorted(checkpoint_paths), dtype=np.int64)

    bases: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    previous_basis: torch.Tensor | None = None
    models: list[Rank2CTRNN] = []
    for epoch in epochs:
        model = Rank2CTRNN(
            model_config, generator=torch.Generator().manual_seed(0)
        )
        payload = load_checkpoint(checkpoint_paths[int(epoch)])
        model.load_state_dict(payload["model_state"])
        model.eval()
        with torch.no_grad():
            prediction, states, _ = model.rollout(inputs)
            basis = _aligned_row_space_basis(model.recurrent_matrix, previous_basis)
            trajectory = states @ basis
        previous_basis = basis
        models.append(model)
        bases.append(basis.cpu().numpy())
        trajectories.append(trajectory.cpu().numpy())
        predictions.append(prediction.cpu().numpy())

    bounds = config.coordinate_bounds or _trajectory_bounds(
        np.stack(trajectories), evaluation_batch.valid_mask, config.padding_fraction
    )
    bounds_source = (
        "explicit" if config.coordinate_bounds is not None else "trajectory"
    )
    expansions = 0
    for _ in range(config.max_bounds_expansions + 1):
        grid = _evaluate_latent_grid(
            models=models,
            bases=bases,
            model_config=model_config,
            input_condition=config.input_condition,
            bounds=bounds,
            grid_points=config.grid_points,
        )
        if config.coordinate_bounds is not None:
            break
        touched = _edge_touched_sides(grid["speed_minima"])
        if not touched or expansions >= config.max_bounds_expansions:
            break
        bounds = _expand_bounds(bounds, touched, config.bounds_expansion_fraction)
        expansions += 1
    flows = grid["flows"]
    speeds = grid["speeds"]
    speed_minima = grid["speed_minima"]
    grid_outputs = grid["grid_outputs"]
    grid_x = grid["grid_x"]
    grid_y = grid["grid_y"]
    if bounds_source == "trajectory" and expansions:
        bounds_source = f"trajectory+edge_minima_expansion({expansions})"

    available_epochs = set(int(epoch) for epoch in epochs)
    representative = tuple(
        epoch for epoch in config.representative_epochs if epoch in available_epochs
    )
    final_epoch = int(epochs[-1])
    if final_epoch not in representative:
        representative = (*representative, final_epoch)

    data_path = output_dir / "latent_dynamics.npz"
    np.savez_compressed(
        data_path,
        epochs=epochs,
        basis=np.stack(bases),
        grid_x=grid_x,
        grid_y=grid_y,
        flow=np.stack(flows),
        speed=np.stack(speeds),
        speed_minimum_mask=np.stack(speed_minima),
        grid_output=np.stack(grid_outputs),
        trajectory=np.stack(trajectories),
        prediction=np.stack(predictions),
        target=evaluation_batch.target,
        valid_mask=evaluation_batch.valid_mask,
        inputs=evaluation_batch.inputs,
        bounds=np.asarray(bounds),
        input_condition=np.asarray(config.input_condition),
    )
    metadata_path = output_dir / "latent_dynamics.yaml"
    dump_yaml(
        {
            "name": config.name,
            "epochs": epochs.tolist(),
            "representative_epochs": list(representative),
            "checkpoint_paths": {
                int(epoch): str(checkpoint_paths[int(epoch)]) for epoch in epochs
            },
            "flow_definition": (
                "tau*dκ/dt = -κ + Q^T tanh(WQκ + W_in*u + b)"
            ),
            "speed_definition": "||tau * F_kappa||_2",
            "input_condition": list(config.input_condition),
            "coordinate_basis": (
                "right-singular row-space basis of W, sequentially aligned by "
                "orthogonal Procrustes"
            ),
            "coordinate_bounds": list(bounds),
            "coordinate_bounds_source": bounds_source,
            "speed_floor": config.speed_floor,
            "speed_minimum_definition": (
                "descriptive grid candidates no larger than their eight neighbors; "
                "not optimized or classified slow points"
            ),
            "evaluation_trials": [asdict(item) for item in config.evaluation_trials],
        },
        metadata_path,
    )
    return LatentDynamicsResult(
        data_path=data_path,
        metadata_path=metadata_path,
        epochs=epochs,
        representative_epochs=representative,
    )


def _evaluation_batch(
    task: IntervalCategorizationTask,
    configurations: Sequence[EvaluationTrialConfig],
) -> TaskBatch:
    trials = [
        task.build_trial(
            interval=item.interval, delay=item.delay, s1_onset=item.s1_onset
        )
        for item in configurations
    ]
    return task.collate_trials(trials)


def _aligned_row_space_basis(
    recurrent_matrix: torch.Tensor, previous: torch.Tensor | None
) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(recurrent_matrix, full_matrices=False)
    basis = vh[:2].T
    if previous is None:
        return _canonicalize_signs(basis)
    left, _, right_t = torch.linalg.svd(basis.T @ previous)
    return basis @ (left @ right_t)


def _canonicalize_signs(basis: torch.Tensor) -> torch.Tensor:
    result = basis.clone()
    for column in range(result.shape[1]):
        pivot = torch.argmax(torch.abs(result[:, column]))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    return result


def _trajectory_bounds(
    trajectories: np.ndarray,
    valid_mask: np.ndarray,
    padding_fraction: float,
) -> tuple[float, float, float, float]:
    valid_points = trajectories[:, valid_mask, :].reshape(-1, 2)
    minima = valid_points.min(axis=0)
    maxima = valid_points.max(axis=0)
    center = (minima + maxima) / 2
    span = np.maximum(maxima - minima, 1.0)
    half_span = span * (0.5 + padding_fraction)
    return (
        float(center[0] - half_span[0]),
        float(center[0] + half_span[0]),
        float(center[1] - half_span[1]),
        float(center[1] + half_span[1]),
    )


def _evaluate_latent_grid(
    *,
    models: Sequence[Rank2CTRNN],
    bases: Sequence[np.ndarray],
    model_config: Rank2CTRNNConfig,
    input_condition: Sequence[float],
    bounds: tuple[float, float, float, float],
    grid_points: int,
) -> dict[str, Any]:
    """Evaluate flow, speed, grid minima, and readout on one aligned grid."""
    x_values = np.linspace(bounds[0], bounds[1], grid_points)
    y_values = np.linspace(bounds[2], bounds[3], grid_points)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    points = torch.as_tensor(
        np.column_stack((grid_x.ravel(), grid_y.ravel())),
        dtype=model_config.torch_dtype,
    )
    input_condition_t = torch.as_tensor(
        input_condition, dtype=model_config.torch_dtype
    ).expand(points.shape[0], -1)

    flows: list[np.ndarray] = []
    speeds: list[np.ndarray] = []
    speed_minima: list[np.ndarray] = []
    grid_outputs: list[np.ndarray] = []
    for model, basis_array in zip(models, bases):
        basis = torch.as_tensor(basis_array, dtype=model_config.torch_dtype)
        with torch.no_grad():
            loading = model.recurrent_matrix @ basis
            drive = (
                points @ loading.T
                + input_condition_t @ model.input_weight.T
                + model.bias
            )
            flow = (-points + torch.tanh(drive) @ basis) / model_config.tau
            normalized_speed = torch.linalg.vector_norm(
                model_config.tau * flow, dim=-1
            )
            latent_readout = basis.T @ model.effective_readout
            grid_output = torch.tanh(
                points @ latent_readout + model.readout_bias
            )[:, 0]
        flows.append(flow.reshape(grid_points, grid_points, 2).numpy())
        speed_grid = normalized_speed.reshape(grid_points, grid_points).numpy()
        speeds.append(speed_grid)
        speed_minima.append(_grid_local_minima(speed_grid))
        grid_outputs.append(
            grid_output.reshape(grid_points, grid_points).numpy()
        )
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "flows": np.stack(flows),
        "speeds": np.stack(speeds),
        "speed_minima": np.stack(speed_minima),
        "grid_outputs": np.stack(grid_outputs),
    }


def _edge_touched_sides(minima: np.ndarray) -> set[str]:
    """Sides whose outermost interior ring holds a grid-minimum candidate.

    Grid minima are only defined on interior samples, so a candidate on the
    ring one cell from the border suggests the slow structure may continue
    past the current plotting bounds.
    """
    rows, columns = np.nonzero(np.any(minima, axis=0))
    touched: set[str] = set()
    if rows.size:
        if (rows == 1).any():
            touched.add("y_min")
        if (rows == minima.shape[1] - 2).any():
            touched.add("y_max")
        if (columns == 1).any():
            touched.add("x_min")
        if (columns == minima.shape[2] - 2).any():
            touched.add("x_max")
    return touched


def _expand_bounds(
    bounds: tuple[float, float, float, float],
    touched: set[str],
    fraction: float,
) -> tuple[float, float, float, float]:
    """Widen the sides flagged by :func:`_edge_touched_sides`."""
    x_min, x_max, y_min, y_max = bounds
    x_margin = (x_max - x_min) * fraction
    y_margin = (y_max - y_min) * fraction
    if "x_min" in touched:
        x_min -= x_margin
    if "x_max" in touched:
        x_max += x_margin
    if "y_min" in touched:
        y_min -= y_margin
    if "y_max" in touched:
        y_max += y_margin
    return (x_min, x_max, y_min, y_max)


def _grid_local_minima(speed: np.ndarray) -> np.ndarray:
    """Mark interior grid samples no faster than all eight neighbors."""
    mask = np.zeros_like(speed, dtype=np.bool_)
    center = speed[1:-1, 1:-1]
    is_minimum = np.ones_like(center, dtype=np.bool_)
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            neighbor = speed[
                1 + row_offset : speed.shape[0] - 1 + row_offset,
                1 + column_offset : speed.shape[1] - 1 + column_offset,
            ]
            is_minimum &= center <= neighbor
    mask[1:-1, 1:-1] = is_minimum
    return mask
