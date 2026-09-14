"""Trajectory-neighborhood diagnostics for rank-K latent systems, K >= 3."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import torch

from slow_manifold.config import dump_yaml
from slow_manifold.models import Rank2CTRNN, Rank2CTRNNConfig
from slow_manifold.tasks import ConfiguredTask, TaskBatch
from slow_manifold.training.checkpoint import load_checkpoint

if TYPE_CHECKING:
    from .latent_dynamics import (
        AnalysisConfig,
        EvaluationTrialConfig,
        LatentDynamicsResult,
    )


def analyze_high_dimensional_checkpoints(
    *,
    checkpoint_paths: Mapping[int, Path],
    model_config: Rank2CTRNNConfig,
    task: ConfiguredTask,
    config: AnalysisConfig,
    output_dir: Path,
) -> LatentDynamicsResult:
    """Sample exact K-D dynamics near tasks and project only for 3-D display."""
    from .latent_dynamics import LatentDynamicsResult

    if model_config.rank < 3:
        raise ValueError("high-dimensional diagnostics require rank >= 3")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_batch = _evaluation_batch(task, config.evaluation_trials)
    trajectory_rng = np.random.default_rng(config.trajectory_seed)
    trajectory_batch = task.generate_batch(config.trajectory_count, trajectory_rng)
    evaluation_inputs = torch.as_tensor(
        evaluation_batch.inputs, dtype=model_config.torch_dtype
    )
    trajectory_inputs = torch.as_tensor(
        trajectory_batch.inputs, dtype=model_config.torch_dtype
    )
    epochs = np.asarray(sorted(checkpoint_paths), dtype=np.int64)

    models: list[Rank2CTRNN] = []
    predictions: list[np.ndarray] = []
    latent_trajectories: list[np.ndarray] = []
    display_trajectories: list[np.ndarray] = []
    projection_means: list[np.ndarray] = []
    projection_components: list[np.ndarray] = []
    explained_variance_ratios: list[np.ndarray] = []
    neighborhood_latent: list[np.ndarray] = []
    neighborhood_display: list[np.ndarray] = []
    neighborhood_flow: list[np.ndarray] = []
    neighborhood_q: list[np.ndarray] = []
    neighborhood_eigenvalues: list[np.ndarray] = []
    neighborhood_spectral_abscissa: list[np.ndarray] = []
    low_q_masks: list[np.ndarray] = []
    near_zero_masks: list[np.ndarray] = []
    low_q_thresholds: list[float] = []
    sampling_radii: list[np.ndarray] = []

    previous_components: torch.Tensor | None = None
    valid_mask_t = torch.as_tensor(trajectory_batch.valid_mask, dtype=torch.bool)
    for epoch in epochs:
        model = Rank2CTRNN(
            model_config, generator=torch.Generator().manual_seed(0)
        )
        payload = load_checkpoint(checkpoint_paths[int(epoch)])
        model.load_state_dict(payload["model_state"])
        model.eval()
        model.requires_grad_(False)
        with torch.no_grad():
            prediction, _, _ = model.rollout(evaluation_inputs)
            _, _, latent = model.rollout(trajectory_inputs)
        valid_latent = latent[valid_mask_t]
        mean, components, explained = _display_projection(
            valid_latent,
            rank=model_config.rank,
            previous_components=previous_components,
        )
        previous_components = components
        display = (latent - mean) @ components

        samples, radius = _sample_trajectory_neighborhood(
            valid_latent,
            epoch=int(epoch),
            seed=config.trajectory_seed,
            anchor_count=config.neighborhood_sampling.anchor_count,
            samples_per_anchor=config.neighborhood_sampling.samples_per_anchor,
            radius_fraction=config.neighborhood_sampling.radius_fraction,
            minimum_radius=config.neighborhood_sampling.minimum_radius,
        )
        sample_inputs = torch.as_tensor(
            config.input_condition, dtype=model_config.torch_dtype
        ).expand(samples.shape[0], -1)
        with torch.no_grad():
            flow = model.latent_flow(samples, sample_inputs)
            jacobian = model.latent_jacobian(samples, sample_inputs)
            eigenvalues = torch.linalg.eigvals(jacobian)
            normalized_flow = model_config.tau * flow
            q = 0.5 * normalized_flow.square().sum(dim=-1)
        spectral_abscissa = eigenvalues.real.max(dim=-1).values
        q_threshold = float(
            torch.quantile(
                q, config.neighborhood_sampling.low_q_quantile
            ).cpu()
        )
        low_q = q <= q_threshold
        near_zero = (
            spectral_abscissa.abs()
            <= config.neighborhood_sampling.near_zero_eigenvalue_tolerance
        )
        low_q = _cap_mask(
            low_q,
            q,
            config.neighborhood_sampling.max_plot_points,
        )
        near_zero = _cap_mask(
            near_zero,
            spectral_abscissa.abs(),
            config.neighborhood_sampling.max_plot_points,
        )

        models.append(model)
        predictions.append(prediction.cpu().numpy())
        latent_trajectories.append(latent.cpu().numpy())
        display_trajectories.append(display.cpu().numpy())
        projection_means.append(mean.cpu().numpy())
        projection_components.append(components.cpu().numpy())
        explained_variance_ratios.append(explained.cpu().numpy())
        neighborhood_latent.append(samples.cpu().numpy())
        neighborhood_display.append(((samples - mean) @ components).cpu().numpy())
        neighborhood_flow.append(flow.cpu().numpy())
        neighborhood_q.append(q.cpu().numpy())
        neighborhood_eigenvalues.append(eigenvalues.cpu().numpy())
        neighborhood_spectral_abscissa.append(spectral_abscissa.cpu().numpy())
        low_q_masks.append(low_q.cpu().numpy())
        near_zero_masks.append(near_zero.cpu().numpy())
        low_q_thresholds.append(q_threshold)
        sampling_radii.append(radius.cpu().numpy())

    trajectory_array = np.stack(display_trajectories)
    display_bounds = _shared_display_bounds(
        trajectory_array,
        trajectory_batch.valid_mask,
        np.stack(neighborhood_display),
        config.padding_fraction,
    )
    coordinate_kind = "exact_kappa" if model_config.rank == 3 else "trajectory_pca"
    coordinate_labels = (
        np.asarray(["kappa_1", "kappa_2", "kappa_3"])
        if model_config.rank == 3
        else np.asarray(["PC1", "PC2", "PC3"])
    )

    data_path = output_dir / "latent_dynamics.npz"
    np.savez_compressed(
        data_path,
        epochs=epochs,
        latent_rank=np.asarray(model_config.rank),
        coordinate_dimension=np.asarray(3),
        coordinate_kind=np.asarray(coordinate_kind),
        coordinate_labels=coordinate_labels,
        projection_mean=np.stack(projection_means),
        projection_components=np.stack(projection_components),
        pca_explained_variance_ratio=np.stack(explained_variance_ratios),
        display_bounds=display_bounds,
        trajectory=trajectory_array,
        trajectory_latent=np.stack(latent_trajectories),
        trajectory_target=trajectory_batch.target,
        trajectory_valid_mask=trajectory_batch.valid_mask,
        trajectory_inputs=trajectory_batch.inputs,
        trajectory_trial_interval=np.asarray(
            [item.interval for item in trajectory_batch.metadata]
        ),
        trajectory_trial_delay=np.asarray(
            [item.delay for item in trajectory_batch.metadata]
        ),
        trajectory_trial_s1_step=np.asarray(
            [item.s1_step for item in trajectory_batch.metadata]
        ),
        trajectory_trial_s2_step=np.asarray(
            [item.s2_step for item in trajectory_batch.metadata]
        ),
        trajectory_trial_go_step=np.asarray(
            [item.go_step for item in trajectory_batch.metadata]
        ),
        trajectory_trial_response_step=np.asarray(
            [item.response_step for item in trajectory_batch.metadata]
        ),
        trajectory_trial_steps=np.asarray(
            [item.trial_steps for item in trajectory_batch.metadata]
        ),
        neighborhood_latent=np.stack(neighborhood_latent),
        neighborhood_coordinates=np.stack(neighborhood_display),
        neighborhood_flow=np.stack(neighborhood_flow),
        neighborhood_q=np.stack(neighborhood_q),
        neighborhood_speed=np.sqrt(2.0 * np.stack(neighborhood_q)),
        neighborhood_jacobian_eigenvalues=np.stack(neighborhood_eigenvalues),
        neighborhood_spectral_abscissa=np.stack(
            neighborhood_spectral_abscissa
        ),
        neighborhood_low_q_mask=np.stack(low_q_masks),
        neighborhood_near_zero_mask=np.stack(near_zero_masks),
        neighborhood_low_q_threshold=np.asarray(low_q_thresholds),
        neighborhood_sampling_radius=np.stack(sampling_radii),
        prediction=np.stack(predictions),
        target=evaluation_batch.target,
        valid_mask=evaluation_batch.valid_mask,
        inputs=evaluation_batch.inputs,
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
        trial_steps=np.asarray([item.trial_steps for item in evaluation_batch.metadata]),
    )

    metadata_path = output_dir / "latent_dynamics.yaml"
    dump_yaml(
        {
            "name": config.name,
            "epochs": epochs.tolist(),
            "checkpoint_paths": {
                int(epoch): str(checkpoint_paths[int(epoch)]) for epoch in epochs
            },
            "latent_rank": model_config.rank,
            "coordinate_dimension": 3,
            "coordinate_kind": coordinate_kind,
            "coordinate_definition": (
                "raw exact kappa=N^T*x; gauge-dependent across checkpoints"
                if model_config.rank == 3
                else (
                    "PC1-3 fitted to exact-kappa task trajectories at each "
                    "checkpoint and sign-aligned to the previous checkpoint; "
                    "display only"
                )
            ),
            "pca_scope": (
                "not used to define dynamics, q, Jacobian, or eigenvalues"
                if model_config.rank > 3
                else "not applicable"
            ),
            "flow_definition": (
                "tau*dκ/dt = -κ + N^T tanh(Mκ + W_in*u + b)"
            ),
            "q_definition": "q=0.5*||tau*F_kappa||_2^2 in full K-D kappa space",
            "jacobian_definition": (
                "J_kappa=tau^-1[-I+N^T diag(1-tanh^2(M*kappa+W_in*u+b)) M]"
            ),
            "jacobian_method": "exact analytic K-dimensional latent Jacobian",
            "spectral_abscissa_definition": "max_i Re(lambda_i(J_kappa))",
            "input_condition": list(config.input_condition),
            "trajectory_sampling": {
                "source": "configured validation task distribution",
                "count": config.trajectory_count,
                "seed": config.trajectory_seed,
                "initial_state": "zero",
                "neural_noise": "none",
            },
            "neighborhood_sampling": {
                **asdict(config.neighborhood_sampling),
                "space": "full exact K-dimensional kappa space",
                "anchors": "uniform samples from valid task-trajectory states",
                "radius": (
                    "per-coordinate max(trajectory_std*radius_fraction, "
                    "minimum_radius)"
                ),
                "low_q_selection": "within-checkpoint empirical q quantile",
                "near_zero_selection": (
                    "abs(max_i Re(lambda_i)) <= near_zero_eigenvalue_tolerance"
                ),
                "interpretation": (
                    "descriptive sampled regions; not fixed points, ghost "
                    "mechanisms, or invariant slow manifolds"
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
        data_path=data_path, metadata_path=metadata_path, epochs=epochs
    )


def _evaluation_batch(
    task: ConfiguredTask, configurations: Sequence[EvaluationTrialConfig]
) -> TaskBatch:
    return task.collate_trials(
        [
            task.build_trial(
                interval=item.interval,
                delay=item.delay,
                s1_onset=item.s1_onset,
            )
            for item in configurations
        ]
    )


def _display_projection(
    valid_latent: torch.Tensor,
    *,
    rank: int,
    previous_components: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rank == 3:
        mean = torch.zeros(rank, dtype=valid_latent.dtype)
        components = torch.eye(rank, dtype=valid_latent.dtype)
        explained = _explained_variance_ratio(valid_latent, components)
        return mean, components, explained

    mean = valid_latent.mean(dim=0)
    centered = valid_latent - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:3].T
    if previous_components is None:
        components = _canonicalize_signs(components)
    else:
        components = components.clone()
        for column in range(components.shape[1]):
            if torch.dot(components[:, column], previous_components[:, column]) < 0:
                components[:, column] *= -1
    denominator = singular_values.square().sum().clamp_min(
        torch.finfo(valid_latent.dtype).eps
    )
    explained = singular_values[:3].square() / denominator
    return mean, components, explained


def _explained_variance_ratio(
    points: torch.Tensor, components: torch.Tensor
) -> torch.Tensor:
    centered = points - points.mean(dim=0)
    total = centered.square().sum().clamp_min(torch.finfo(points.dtype).eps)
    return (centered @ components).square().sum(dim=0) / total


def _canonicalize_signs(components: torch.Tensor) -> torch.Tensor:
    result = components.clone()
    for column in range(result.shape[1]):
        pivot = torch.argmax(result[:, column].abs())
        if result[pivot, column] < 0:
            result[:, column] *= -1
    return result


def _sample_trajectory_neighborhood(
    valid_latent: torch.Tensor,
    *,
    epoch: int,
    seed: int,
    anchor_count: int,
    samples_per_anchor: int,
    radius_fraction: float,
    minimum_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])
    )
    indices = torch.randint(
        valid_latent.shape[0], (anchor_count,), generator=generator
    )
    anchors = valid_latent[indices]
    radius = torch.clamp(
        valid_latent.std(dim=0, unbiased=False) * radius_fraction,
        min=minimum_radius,
    )
    if samples_per_anchor == 0:
        return anchors, radius
    noise = torch.randn(
        anchor_count,
        samples_per_anchor,
        valid_latent.shape[1],
        dtype=valid_latent.dtype,
        generator=generator,
    )
    perturbed = anchors[:, None, :] + noise * radius
    samples = torch.cat((anchors[:, None, :], perturbed), dim=1)
    return samples.reshape(-1, valid_latent.shape[1]), radius


def _cap_mask(
    mask: torch.Tensor, score: torch.Tensor, maximum: int
) -> torch.Tensor:
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    if selected.numel() <= maximum:
        return mask
    keep = selected[torch.argsort(score[selected])[:maximum]]
    result = torch.zeros_like(mask)
    result[keep] = True
    return result


def _shared_display_bounds(
    trajectories: np.ndarray,
    valid_mask: np.ndarray,
    neighborhoods: np.ndarray,
    padding_fraction: float,
) -> np.ndarray:
    valid = trajectories[:, valid_mask, :].reshape(-1, 3)
    points = np.concatenate((valid, neighborhoods.reshape(-1, 3)), axis=0)
    minima = points.min(axis=0)
    maxima = points.max(axis=0)
    center = 0.5 * (minima + maxima)
    span = np.maximum(maxima - minima, 1.0e-6)
    half_span = 0.5 * span * (1.0 + 2.0 * padding_fraction)
    return np.column_stack((center - half_span, center + half_span))
