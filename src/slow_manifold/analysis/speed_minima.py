"""Dinc-style refinement and classification of latent speed minima."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from slow_manifold.models import Rank2CTRNN


class SpeedMinimumConfigError(ValueError):
    """Raised when speed-minimum classification settings are invalid."""


@dataclass(frozen=True)
class SpeedMinimumClassificationConfig:
    """Numerical tolerances for Dinc-style speed-minimum classification."""

    max_iterations: int = 75
    optimizer_gradient_tolerance: float = 1.0e-8
    parameter_tolerance: float = 1.0e-12
    stationarity_tolerance: float = 1.0e-6
    fixed_speed_tolerance: float = 1.0e-6
    hessian_eigenvalue_tolerance: float = 1.0e-8
    zero_eigenvalue_tolerance: float = 1.0e-2
    transverse_stability_tolerance: float = 1.0e-4
    deduplication_tolerance: float = 1.0e-3

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "SpeedMinimumClassificationConfig":
        if data is not None and not isinstance(data, Mapping):
            raise SpeedMinimumConfigError(
                "speed_minimum_classification must be a mapping"
            )
        raw = {} if data is None else data
        try:
            config = cls(
                max_iterations=int(raw.get("max_iterations", 75)),
                optimizer_gradient_tolerance=float(
                    raw.get("optimizer_gradient_tolerance", 1.0e-8)
                ),
                parameter_tolerance=float(raw.get("parameter_tolerance", 1.0e-12)),
                stationarity_tolerance=float(
                    raw.get("stationarity_tolerance", 1.0e-6)
                ),
                fixed_speed_tolerance=float(
                    raw.get("fixed_speed_tolerance", 1.0e-6)
                ),
                hessian_eigenvalue_tolerance=float(
                    raw.get("hessian_eigenvalue_tolerance", 1.0e-8)
                ),
                zero_eigenvalue_tolerance=float(
                    raw.get("zero_eigenvalue_tolerance", 1.0e-2)
                ),
                transverse_stability_tolerance=float(
                    raw.get("transverse_stability_tolerance", 1.0e-4)
                ),
                deduplication_tolerance=float(
                    raw.get("deduplication_tolerance", 1.0e-3)
                ),
            )
        except (TypeError, ValueError) as error:
            raise SpeedMinimumConfigError(
                f"Invalid speed-minimum classification config: {error}"
            ) from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_iterations <= 0:
            raise SpeedMinimumConfigError("max_iterations must be positive")
        tolerances = (
            self.optimizer_gradient_tolerance,
            self.parameter_tolerance,
            self.stationarity_tolerance,
            self.fixed_speed_tolerance,
            self.hessian_eigenvalue_tolerance,
            self.zero_eigenvalue_tolerance,
            self.transverse_stability_tolerance,
            self.deduplication_tolerance,
        )
        if any(value <= 0 for value in tolerances):
            raise SpeedMinimumConfigError("classification tolerances must be positive")


@dataclass(frozen=True)
class ClassifiedSpeedMinima:
    """Structured refined candidates for one checkpoint."""

    coordinates: NDArray[np.floating]
    flow: NDArray[np.floating]
    speed: NDArray[np.floating]
    q: NDArray[np.floating]
    jacobian: NDArray[np.floating]
    eigenvalues: NDArray[np.complexfloating]
    q_gradient_norm: NDArray[np.floating]
    q_hessian_eigenvalues: NDArray[np.floating]
    optimization_converged: NDArray[np.bool_]
    is_local_minimum: NDArray[np.bool_]
    is_attractor: NDArray[np.bool_]
    classification: NDArray[np.str_]
    source_count: NDArray[np.integer]


def refine_and_classify_speed_minima(
    *,
    model: Rank2CTRNN,
    basis: torch.Tensor,
    grid_x: NDArray[np.floating],
    grid_y: NDArray[np.floating],
    grid_minimum_mask: NDArray[np.bool_],
    input_condition: Sequence[float],
    bounds: tuple[float, float, float, float],
    config: SpeedMinimumClassificationConfig,
) -> ClassifiedSpeedMinima:
    """Refine grid candidates by minimizing ``q=0.5*||tau*F||^2``.

    Classification follows Dinc's hierarchy: zero-flow minima are fixed
    points; nonzero local minima are slow points; a slow point with one
    effective zero mode and stable remaining modes is a latent ghost candidate.
    """
    rows, columns = np.nonzero(grid_minimum_mask)
    if rows.size == 0:
        return _empty_result(model.config.rank)
    model.to(dtype=torch.float64)
    basis = basis.to(dtype=torch.float64)
    seeds = np.column_stack((grid_x[rows, columns], grid_y[rows, columns]))
    points = torch.as_tensor(
        seeds, dtype=basis.dtype, device=basis.device
    ).clone()
    points.requires_grad_(True)
    inputs = torch.as_tensor(
        input_condition, dtype=basis.dtype, device=basis.device
    ).expand(points.shape[0], -1)

    optimizer = torch.optim.LBFGS(
        [points],
        lr=1.0,
        max_iter=config.max_iterations,
        tolerance_grad=config.optimizer_gradient_tolerance,
        tolerance_change=config.parameter_tolerance,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        normalized_flow = model.config.tau * model.row_space_flow(
            points, inputs, basis
        )
        total_q = 0.5 * normalized_flow.square().sum()
        total_q.backward()
        return total_q

    optimizer.step(closure)
    refined, source_counts = _deduplicate_points(
        points.detach(), model, basis, inputs[0], config.deduplication_tolerance
    )
    return _classify_points(
        refined,
        source_counts,
        model=model,
        basis=basis,
        input_condition=inputs[0],
        bounds=bounds,
        config=config,
    )


def _deduplicate_points(
    points: torch.Tensor,
    model: Rank2CTRNN,
    basis: torch.Tensor,
    input_condition: torch.Tensor,
    tolerance: float,
) -> tuple[torch.Tensor, NDArray[np.int64]]:
    with torch.no_grad():
        inputs = input_condition.expand(points.shape[0], -1)
        normalized_flow = model.config.tau * model.row_space_flow(
            points, inputs, basis
        )
        q = 0.5 * normalized_flow.square().sum(dim=-1)
    order = torch.argsort(q)
    unique: list[torch.Tensor] = []
    counts: list[int] = []
    for index in order:
        point = points[index]
        match = next(
            (
                candidate_index
                for candidate_index, candidate in enumerate(unique)
                if torch.linalg.vector_norm(point - candidate) <= tolerance
            ),
            None,
        )
        if match is None:
            unique.append(point)
            counts.append(1)
        else:
            counts[match] += 1
    return torch.stack(unique), np.asarray(counts, dtype=np.int64)


def _classify_points(
    points: torch.Tensor,
    source_counts: NDArray[np.int64],
    *,
    model: Rank2CTRNN,
    basis: torch.Tensor,
    input_condition: torch.Tensor,
    bounds: tuple[float, float, float, float],
    config: SpeedMinimumClassificationConfig,
) -> ClassifiedSpeedMinima:
    records: dict[str, list[Any]] = {
        "coordinates": [],
        "flow": [],
        "speed": [],
        "q": [],
        "jacobian": [],
        "eigenvalues": [],
        "q_gradient_norm": [],
        "q_hessian_eigenvalues": [],
        "optimization_converged": [],
        "is_local_minimum": [],
        "is_attractor": [],
        "classification": [],
    }
    for point in points:
        coordinate = point.detach().clone().requires_grad_(True)

        def q_function(value: torch.Tensor) -> torch.Tensor:
            normalized_flow = model.config.tau * model.row_space_flow(
                value.unsqueeze(0), input_condition.unsqueeze(0), basis
            )[0]
            return 0.5 * normalized_flow.square().sum()

        q = q_function(coordinate)
        gradient = torch.autograd.grad(q, coordinate, create_graph=False)[0]
        hessian = torch.autograd.functional.hessian(q_function, coordinate)
        with torch.no_grad():
            flow = model.row_space_flow(
                coordinate.unsqueeze(0), input_condition.unsqueeze(0), basis
            )[0]
            normalized_speed = torch.linalg.vector_norm(model.config.tau * flow)
            jacobian = model.row_space_jacobian(
                coordinate.unsqueeze(0), input_condition.unsqueeze(0), basis
            )[0]
            eigenvalues = torch.linalg.eigvals(jacobian)
            hessian_eigenvalues = torch.linalg.eigvalsh(hessian)

        gradient_norm = float(torch.linalg.vector_norm(gradient))
        speed = float(normalized_speed)
        inside = _inside_bounds(coordinate, bounds)
        converged = gradient_norm <= config.stationarity_tolerance
        local_minimum = bool(
            converged
            and inside
            and float(hessian_eigenvalues.min())
            > config.hessian_eigenvalue_tolerance
        )
        attracting = bool(
            torch.all(
                eigenvalues.real < -config.transverse_stability_tolerance
            )
        )
        classification = classify_speed_minimum(
            speed=speed,
            eigenvalues=eigenvalues.cpu().numpy(),
            is_local_minimum=local_minimum,
            config=config,
        )

        records["coordinates"].append(coordinate.detach().cpu().numpy())
        records["flow"].append(flow.cpu().numpy())
        records["speed"].append(speed)
        records["q"].append(float(q.detach()))
        records["jacobian"].append(jacobian.cpu().numpy())
        records["eigenvalues"].append(eigenvalues.cpu().numpy())
        records["q_gradient_norm"].append(gradient_norm)
        records["q_hessian_eigenvalues"].append(
            hessian_eigenvalues.cpu().numpy()
        )
        records["optimization_converged"].append(converged)
        records["is_local_minimum"].append(local_minimum)
        records["is_attractor"].append(attracting)
        records["classification"].append(classification)

    return ClassifiedSpeedMinima(
        coordinates=np.asarray(records["coordinates"]),
        flow=np.asarray(records["flow"]),
        speed=np.asarray(records["speed"]),
        q=np.asarray(records["q"]),
        jacobian=np.asarray(records["jacobian"]),
        eigenvalues=np.asarray(records["eigenvalues"]),
        q_gradient_norm=np.asarray(records["q_gradient_norm"]),
        q_hessian_eigenvalues=np.asarray(records["q_hessian_eigenvalues"]),
        optimization_converged=np.asarray(
            records["optimization_converged"], dtype=np.bool_
        ),
        is_local_minimum=np.asarray(records["is_local_minimum"], dtype=np.bool_),
        is_attractor=np.asarray(records["is_attractor"], dtype=np.bool_),
        classification=np.asarray(records["classification"], dtype="<U24"),
        source_count=source_counts,
    )


def classify_speed_minimum(
    *,
    speed: float,
    eigenvalues: Sequence[complex] | NDArray[np.complexfloating],
    is_local_minimum: bool,
    config: SpeedMinimumClassificationConfig,
) -> str:
    """Classify one refined candidate from its speed and local spectrum."""
    if not is_local_minimum:
        return "unresolved"
    if speed <= config.fixed_speed_tolerance:
        return "fixed_point"
    spectrum = np.asarray(eigenvalues)
    zero_index = int(np.argmin(np.abs(spectrum)))
    zero_mode = bool(
        np.abs(spectrum[zero_index]) <= config.zero_eigenvalue_tolerance
    )
    transverse = np.delete(spectrum, zero_index)
    transverse_stable = bool(
        np.all(transverse.real < -config.transverse_stability_tolerance)
    )
    if zero_mode and transverse_stable:
        return "latent_ghost_candidate"
    return "slow_point"


def _inside_bounds(
    point: torch.Tensor, bounds: tuple[float, float, float, float]
) -> bool:
    x_min, x_max, y_min, y_max = bounds
    detached = point.detach()
    return bool(
        x_min <= float(detached[0]) <= x_max
        and y_min <= float(detached[1]) <= y_max
    )


def _empty_result(rank: int) -> ClassifiedSpeedMinima:
    return ClassifiedSpeedMinima(
        coordinates=np.empty((0, rank), dtype=np.float32),
        flow=np.empty((0, rank), dtype=np.float32),
        speed=np.empty(0, dtype=np.float32),
        q=np.empty(0, dtype=np.float32),
        jacobian=np.empty((0, rank, rank), dtype=np.float32),
        eigenvalues=np.empty((0, rank), dtype=np.complex64),
        q_gradient_norm=np.empty(0, dtype=np.float32),
        q_hessian_eigenvalues=np.empty((0, rank), dtype=np.float32),
        optimization_converged=np.empty(0, dtype=np.bool_),
        is_local_minimum=np.empty(0, dtype=np.bool_),
        is_attractor=np.empty(0, dtype=np.bool_),
        classification=np.empty(0, dtype="<U24"),
        source_count=np.empty(0, dtype=np.int64),
    )
