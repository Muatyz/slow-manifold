"""Low-rank latent dynamics across training checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from slow_manifold.config import dump_yaml
from slow_manifold.models import Rank2CTRNN, Rank2CTRNNConfig
from slow_manifold.tasks import ConfiguredTask, TaskBatch
from slow_manifold.training.checkpoint import load_checkpoint

from .checkpoint_selection import RepresentativeSelectionConfig
from .speed_minima import (
    ClassifiedSpeedMinima,
    SpeedMinimumClassificationConfig,
    refine_and_classify_speed_minima,
)


class AnalysisConfigError(ValueError):
    """Raised when training-dynamics analysis configuration is invalid."""


def _configuration_bool(
    data: Mapping[str, Any], key: str, *, default: bool
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be boolean")
    return value


@dataclass(frozen=True)
class EvaluationTrialConfig:
    interval: float
    delay: float
    s1_onset: float


@dataclass(frozen=True)
class NeighborhoodSamplingConfig:
    """Trajectory-neighborhood sampling used when latent rank is at least 3."""

    anchor_count: int = 64
    samples_per_anchor: int = 8
    radius_fraction: float = 0.05
    minimum_radius: float = 1.0e-3
    low_q_quantile: float = 0.05
    near_zero_eigenvalue_tolerance: float = 1.0e-2
    max_plot_points: int = 512

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "NeighborhoodSamplingConfig":
        values = {} if data is None else dict(data)
        config = cls(
            anchor_count=int(values.get("anchor_count", cls.anchor_count)),
            samples_per_anchor=int(
                values.get("samples_per_anchor", cls.samples_per_anchor)
            ),
            radius_fraction=float(
                values.get("radius_fraction", cls.radius_fraction)
            ),
            minimum_radius=float(
                values.get("minimum_radius", cls.minimum_radius)
            ),
            low_q_quantile=float(
                values.get("low_q_quantile", cls.low_q_quantile)
            ),
            near_zero_eigenvalue_tolerance=float(
                values.get(
                    "near_zero_eigenvalue_tolerance",
                    cls.near_zero_eigenvalue_tolerance,
                )
            ),
            max_plot_points=int(
                values.get("max_plot_points", cls.max_plot_points)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.anchor_count <= 0 or self.samples_per_anchor < 0:
            raise AnalysisConfigError(
                "neighborhood anchor_count must be positive and "
                "samples_per_anchor non-negative"
            )
        if self.radius_fraction < 0 or self.minimum_radius <= 0:
            raise AnalysisConfigError("neighborhood sampling radii are invalid")
        if not 0 < self.low_q_quantile <= 1:
            raise AnalysisConfigError("low_q_quantile must lie in (0, 1]")
        if self.near_zero_eigenvalue_tolerance < 0 or self.max_plot_points <= 0:
            raise AnalysisConfigError(
                "near-zero tolerance must be non-negative and max_plot_points positive"
            )


@dataclass(frozen=True)
class AnalysisConfig:
    name: str
    input_condition: tuple[float, ...]
    representative_selection: RepresentativeSelectionConfig
    grid_points: int
    coordinate_bounds: tuple[float, float, float, float] | None
    square_coordinate_bounds: bool
    padding_fraction: float
    bounds_expansion_fraction: float
    max_bounds_expansions: int
    speed_floor: float
    speed_minimum_classification: SpeedMinimumClassificationConfig
    evaluation_trials: tuple[EvaluationTrialConfig, ...]
    trajectory_count: int
    trajectory_seed: int
    neighborhood_sampling: NeighborhoodSamplingConfig

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
            selection_mapping = data.get("representative_selection")
            # A frozen historical run may still carry the former fixed list.
            # Treat it as an explicit manual selection without keeping that
            # presentation concern in the latent-dynamics implementation.
            if selection_mapping is None and "representative_epochs" in data:
                selection_mapping = {
                    "mode": "manual",
                    "manual_epochs": data["representative_epochs"],
                }
            config = cls(
                name=str(data["name"]),
                input_condition=tuple(float(value) for value in data["input_condition"]),
                representative_selection=RepresentativeSelectionConfig.from_mapping(
                    selection_mapping
                ),
                grid_points=int(data["grid_points"]),
                coordinate_bounds=bounds,  # type: ignore[arg-type]
                square_coordinate_bounds=_configuration_bool(
                    data, "square_coordinate_bounds", default=True
                ),
                padding_fraction=float(data["padding_fraction"]),
                bounds_expansion_fraction=float(
                    data.get("bounds_expansion_fraction", 0.5)
                ),
                max_bounds_expansions=int(data.get("max_bounds_expansions", 2)),
                speed_floor=float(data["speed_floor"]),
                speed_minimum_classification=(
                    SpeedMinimumClassificationConfig.from_mapping(
                        data.get("speed_minimum_classification")
                    )
                ),
                evaluation_trials=trials,
                trajectory_count=int(data.get("trajectory_count", 128)),
                trajectory_seed=int(data.get("trajectory_seed", 20260914)),
                neighborhood_sampling=NeighborhoodSamplingConfig.from_mapping(
                    data.get("neighborhood_sampling")
                ),
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
        if self.trajectory_count <= 0:
            raise AnalysisConfigError("trajectory_count must be positive")
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


def analyze_checkpoints(
    *,
    checkpoint_paths: Mapping[int, Path],
    model_config: Rank2CTRNNConfig,
    task: ConfiguredTask,
    config: AnalysisConfig,
    output_dir: Path,
) -> LatentDynamicsResult:
    """Evaluate exact 2-D fields or trajectory-neighborhood K-D dynamics."""
    if len(config.input_condition) != model_config.input_size:
        raise AnalysisConfigError("input_condition length must match model input_size")
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty")
    if model_config.rank >= 3:
        from .high_dimensional_dynamics import analyze_high_dimensional_checkpoints

        return analyze_high_dimensional_checkpoints(
            checkpoint_paths=checkpoint_paths,
            model_config=model_config,
            task=task,
            config=config,
            output_dir=output_dir,
        )

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
        model.requires_grad_(False)
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
    if config.square_coordinate_bounds:
        bounds = _square_bounds(bounds)
        bounds_source += "+square"
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
        if config.square_coordinate_bounds:
            bounds = _square_bounds(bounds)
        expansions += 1
    flows = grid["flows"]
    speeds = grid["speeds"]
    speed_minima = grid["speed_minima"]
    grid_outputs = grid["grid_outputs"]
    jacobian_eigenvalues = grid["jacobian_eigenvalues"]
    jacobian_spectral_abscissa = grid["jacobian_spectral_abscissa"]
    grid_x = grid["grid_x"]
    grid_y = grid["grid_y"]
    classified_minima = [
        refine_and_classify_speed_minima(
            model=model,
            basis=torch.as_tensor(basis, dtype=model_config.torch_dtype),
            grid_x=grid_x,
            grid_y=grid_y,
            grid_minimum_mask=minimum_mask,
            input_condition=config.input_condition,
            bounds=bounds,
            config=config.speed_minimum_classification,
        )
        for model, basis, minimum_mask in zip(models, bases, speed_minima)
    ]
    minimum_data = _combine_classified_minima(classified_minima)
    if expansions:
        bounds_source += f"+edge_minima_expansion({expansions})"

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
        **minimum_data,
        grid_output=np.stack(grid_outputs),
        jacobian_eigenvalues=jacobian_eigenvalues,
        jacobian_spectral_abscissa=jacobian_spectral_abscissa,
        trajectory=np.stack(trajectories),
        prediction=np.stack(predictions),
        target=evaluation_batch.target,
        valid_mask=evaluation_batch.valid_mask,
        inputs=evaluation_batch.inputs,
        bounds=np.asarray(bounds),
        square_coordinate_bounds=np.asarray(config.square_coordinate_bounds),
        input_condition=np.asarray(config.input_condition),
        task_name=np.asarray(task.config.name),
        trial_interval=np.asarray([item.interval for item in evaluation_batch.metadata]),
        trial_delay=np.asarray([item.delay for item in evaluation_batch.metadata]),
        trial_s1_step=np.asarray([item.s1_step for item in evaluation_batch.metadata]),
        trial_s2_step=np.asarray([item.s2_step for item in evaluation_batch.metadata]),
        trial_go_step=np.asarray([item.go_step for item in evaluation_batch.metadata]),
        trial_response_step=np.asarray(
            [item.response_step for item in evaluation_batch.metadata]
        ),
        trial_steps=np.asarray(
            [item.trial_steps for item in evaluation_batch.metadata]
        ),
    )
    metadata_path = output_dir / "latent_dynamics.yaml"
    dump_yaml(
        {
            "name": config.name,
            "epochs": epochs.tolist(),
            "checkpoint_paths": {
                int(epoch): str(checkpoint_paths[int(epoch)]) for epoch in epochs
            },
            "flow_definition": (
                "tau*dκ/dt = -κ + Q^T tanh(WQκ + W_in*u + b)"
            ),
            "speed_definition": "||tau * F_kappa||_2",
            "jacobian_definition": (
                "J_kappa = tau^-1[-I + Q^T diag(1-tanh^2(WQ*kappa "
                "+ W_in*u + b)) WQ]"
            ),
            "jacobian_method": (
                "exact analytic continuous-time row-space Jacobian"
            ),
            "jacobian_eigenvalue_units": "inverse configured time unit",
            "spectral_abscissa_definition": (
                "max_i Re(lambda_i(J_kappa)); sampled on the full aligned grid"
            ),
            "spectral_abscissa_scope": (
                "descriptive plane map, not a slow-point, ghost, or "
                "bifurcation classification"
            ),
            "input_condition": list(config.input_condition),
            "trajectory_initial_state": "zero",
            "trajectory_neural_noise": "none",
            "vector_field_noise_semantics": "deterministic drift; no sampled noise",
            "coordinate_basis": (
                "right-singular row-space basis of W, sequentially aligned by "
                "orthogonal Procrustes"
            ),
            "coordinate_bounds": list(bounds),
            "coordinate_bounds_source": bounds_source,
            "square_coordinate_bounds": config.square_coordinate_bounds,
            "grid_points": config.grid_points,
            "speed_floor": config.speed_floor,
            "speed_minimum_definition": (
                "eight-neighbor grid candidates refined by minimizing "
                "q=0.5*||tau*F_kappa||^2"
            ),
            "speed_minimum_classification": {
                **asdict(config.speed_minimum_classification),
                "fixed_point": (
                    "local q minimum with normalized speed no greater than "
                    "fixed_speed_tolerance; is_attractor additionally requires "
                    "all eigenvalue real parts below negative stability tolerance"
                ),
                "slow_point": (
                    "nonzero local q minimum that does not meet the ghost criterion"
                ),
                "latent_ghost_candidate": (
                    "nonzero local q minimum with one effective zero eigenvalue "
                    "and all remaining latent modes transversely stable"
                ),
                "scope": (
                    "Dinc-style exact rank-2 latent classification; ghost remains "
                    "a candidate until task relevance and cross-run evidence are checked"
                ),
            },
            "evaluation_trials": [asdict(item) for item in config.evaluation_trials],
            "trajectory_phase_definition": {
                "interval_encoding": "[S1, S2)",
                "delay": "[S2, Go)",
                "post_go_timing": "[Go, target response onset)",
                "response": "[target response onset, trial end)",
            },
        },
        metadata_path,
    )
    return LatentDynamicsResult(
        data_path=data_path,
        metadata_path=metadata_path,
        epochs=epochs,
    )


def _combine_classified_minima(
    results: Sequence[ClassifiedSpeedMinima],
) -> dict[str, np.ndarray]:
    counts = np.asarray([result.coordinates.shape[0] for result in results])
    epoch_index = np.repeat(np.arange(len(results), dtype=np.int64), counts)

    def concatenate(name: str) -> np.ndarray:
        arrays = [getattr(result, name) for result in results]
        return np.concatenate(arrays, axis=0)

    return {
        "speed_minimum_epoch_index": epoch_index,
        "speed_minimum_coordinates": concatenate("coordinates"),
        "speed_minimum_flow": concatenate("flow"),
        "speed_minimum_speed": concatenate("speed"),
        "speed_minimum_q": concatenate("q"),
        "speed_minimum_jacobian": concatenate("jacobian"),
        "speed_minimum_eigenvalues": concatenate("eigenvalues"),
        "speed_minimum_q_gradient_norm": concatenate("q_gradient_norm"),
        "speed_minimum_q_hessian_eigenvalues": concatenate(
            "q_hessian_eigenvalues"
        ),
        "speed_minimum_optimization_converged": concatenate(
            "optimization_converged"
        ),
        "speed_minimum_is_local_minimum": concatenate("is_local_minimum"),
        "speed_minimum_is_attractor": concatenate("is_attractor"),
        "speed_minimum_classification": concatenate("classification"),
        "speed_minimum_source_count": concatenate("source_count"),
    }


def _evaluation_batch(
    task: ConfiguredTask,
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


def _square_bounds(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Expand the shorter coordinate span around its center to form a square."""
    x_min, x_max, y_min, y_max = bounds
    span = max(x_max - x_min, y_max - y_min)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_span = 0.5 * span
    return (
        x_center - half_span,
        x_center + half_span,
        y_center - half_span,
        y_center + half_span,
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
    """Evaluate flow, spectrum, speed minima, and readout on one aligned grid."""
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
    jacobian_eigenvalues: list[np.ndarray] = []
    jacobian_spectral_abscissa: list[np.ndarray] = []
    for model, basis_array in zip(models, bases):
        basis = torch.as_tensor(basis_array, dtype=model_config.torch_dtype)
        with torch.no_grad():
            flow = model.row_space_flow(points, input_condition_t, basis)
            jacobian = model.row_space_jacobian(
                points, input_condition_t, basis
            )
            eigenvalues = torch.linalg.eigvals(jacobian)
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
        eigenvalue_grid = eigenvalues.reshape(
            grid_points, grid_points, model_config.rank
        ).numpy()
        jacobian_eigenvalues.append(eigenvalue_grid)
        jacobian_spectral_abscissa.append(
            eigenvalue_grid.real.max(axis=-1)
        )
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "flows": np.stack(flows),
        "speeds": np.stack(speeds),
        "speed_minima": np.stack(speed_minima),
        "grid_outputs": np.stack(grid_outputs),
        "jacobian_eigenvalues": np.stack(jacobian_eigenvalues),
        "jacobian_spectral_abscissa": np.stack(jacobian_spectral_abscissa),
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
