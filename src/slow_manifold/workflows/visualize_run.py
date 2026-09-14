"""Re-render figures of an existing training run without retraining.

The training workflow already decouples data (``diagnostics/``,
``metrics.csv``) from figures (``figures/``).  This workflow lets a user
re-run only the analysis/visualization stage against a completed run, e.g. to
change the vector-field arrow density, widen the coordinate bounds, or add
representative epochs after the run finished.
"""

from __future__ import annotations

import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from slow_manifold.analysis import (
    AnalysisConfig,
    analyze_checkpoints,
    select_representative_checkpoints,
)
from slow_manifold.config import ConfigError, load_yaml, resolve_experiment
from slow_manifold.models import Rank2CTRNNConfig
from slow_manifold.tasks import PhaseNormalizedLossConfig, create_task
from slow_manifold.training.checkpoint import load_checkpoint
from slow_manifold.utils import get_logger, setup_run_logging
from slow_manifold.visualization import (
    plot_representative_outputs,
    plot_training_curves,
    render_latent_dynamics_collection,
    render_latent_jacobian_collection,
    render_latent_vector_field_collection,
    resolve_dpi_settings,
    resolve_render_movies,
)
from slow_manifold.workflows._stage_config import (
    analysis_fingerprint,
    load_stage_component,
    visualization_fingerprint,
    write_analysis_stage_config,
    write_visualization_stage_config,
)

_EPOCH_CHECKPOINT = re.compile(r"epoch-(\d{6})\.pt")


def rerun_rank2_visualization(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    coordinate_bounds: Sequence[float] | None = None,
    grid_points: int | None = None,
    square_coordinate_bounds: bool | None = None,
    representative_epochs: Sequence[int] | None = None,
    arrow_stride: int | None = None,
    arrow_length_fraction: float | None = None,
    arrow_width: float | None = None,
    trajectory_line_width: float | None = None,
    snapshot_dpi: int | None = None,
    movie_dpi: int | None = None,
    render_movies: bool | None = None,
    config_source: str = "current",
    experiment: str | Path | None = None,
) -> Path:
    """Recompute diagnostics if needed and re-render figures for one run.

    The frozen run supplies task, model, metrics, and checkpoints. By default,
    the run's experiment recipe is resolved again for current analysis and
    visualization settings. ``config_source`` can instead select the original
    launch settings or the latest successful stage settings. Diagnostics are
    recomputed only when their resolved fingerprint changes. Figures are
    written to ``output_dir`` (default: the run's ``figures/``).
    """
    run_path = Path(run_dir).resolve()
    config_path = run_path / "config.yaml"
    metrics_path = run_path / "metrics.csv"
    diagnostics_dir = run_path / "diagnostics"
    data_path = diagnostics_dir / "latent_dynamics.npz"
    if not config_path.is_file():
        raise FileNotFoundError(f"Not a training run (no config.yaml): {run_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Run has no metrics.csv: {run_path}")

    setup_run_logging(run_path)
    logger = get_logger("visualization")
    logger.info("visualization rerun started run=%s", run_path)
    started = time.perf_counter()

    resolved = load_yaml(config_path)
    components = resolved.get("components")
    if not isinstance(components, Mapping):
        raise ConfigError(f"Run config has no resolved components: {config_path}")
    model_mapping = components["model"]
    model_config = Rank2CTRNNConfig.from_mapping(model_mapping)
    task_mapping = components["task"]
    response_threshold = float(
        task_mapping.get("evaluation", {}).get("response_threshold", 0.5)
    )
    (
        analysis_mapping,
        visualization_mapping,
        analysis_source_experiment,
        visualization_source_experiment,
    ) = _resolve_stage_mappings(
        resolved,
        run_path,
        config_source=config_source,
        experiment=experiment,
    )
    analysis_overrides: dict[str, Any] = {}
    visualization_overrides: dict[str, Any] = {}
    if coordinate_bounds is not None:
        analysis_mapping["coordinate_bounds"] = list(coordinate_bounds)
        analysis_overrides["coordinate_bounds"] = list(coordinate_bounds)
    if grid_points is not None:
        analysis_mapping["grid_points"] = grid_points
        analysis_overrides["grid_points"] = grid_points
    if square_coordinate_bounds is not None:
        analysis_mapping["square_coordinate_bounds"] = square_coordinate_bounds
        analysis_overrides["square_coordinate_bounds"] = square_coordinate_bounds
    if arrow_stride is not None:
        visualization_mapping["arrow_stride"] = arrow_stride
        visualization_overrides["arrow_stride"] = arrow_stride
    if arrow_length_fraction is not None:
        visualization_mapping["arrow_length_fraction"] = arrow_length_fraction
        visualization_overrides["arrow_length_fraction"] = arrow_length_fraction
    if arrow_width is not None:
        visualization_mapping["arrow_width"] = arrow_width
        visualization_overrides["arrow_width"] = arrow_width
    if trajectory_line_width is not None:
        visualization_mapping["trajectory_line_width"] = trajectory_line_width
        visualization_overrides["trajectory_line_width"] = trajectory_line_width
    if snapshot_dpi is not None:
        visualization_mapping["snapshot_dpi"] = snapshot_dpi
        visualization_overrides["snapshot_dpi"] = snapshot_dpi
    if movie_dpi is not None:
        visualization_mapping["movie_dpi"] = movie_dpi
        visualization_overrides["movie_dpi"] = movie_dpi
    if render_movies is not None:
        visualization_mapping["render_movies"] = render_movies
        visualization_overrides["render_movies"] = render_movies
    if representative_epochs is not None:
        manual_epochs = [int(epoch) for epoch in representative_epochs]
        selection_mapping = dict(
            analysis_mapping.get("representative_selection", {})
        )
        selection_mapping.update(
            {"mode": "manual", "manual_epochs": manual_epochs}
        )
        analysis_mapping.pop("representative_epochs", None)
        analysis_mapping["representative_selection"] = selection_mapping
        analysis_overrides["representative_selection"] = {
            "mode": "manual",
            "manual_epochs": manual_epochs,
        }
    analysis_config = AnalysisConfig.from_mapping(analysis_mapping)

    checkpoint_paths = _epoch_checkpoint_paths(run_path)
    if analysis_config.representative_selection.mode == "manual":
        checkpoint_paths.update(
            _requested_named_checkpoint_paths(
                run_path,
                analysis_config.representative_selection.manual_epochs,
                checkpoint_paths,
            )
        )
        checkpoint_paths = dict(sorted(checkpoint_paths.items()))
    checkpoint_epochs = (
        list(checkpoint_paths)
        if checkpoint_paths
        else _stored_diagnostic_epochs(data_path)
    )
    desired_analysis_fingerprint = analysis_fingerprint(
        task=task_mapping,
        model=model_mapping,
        analysis=analysis_mapping,
        checkpoint_epochs=checkpoint_epochs,
    )
    stage_config_path = diagnostics_dir / "config.yaml"
    stored_analysis_fingerprint = _stored_latent_analysis_fingerprint(
        stage_config_path,
        task=task_mapping,
        model=model_mapping,
    )
    if stored_analysis_fingerprint is None:
        original_analysis_fingerprint = analysis_fingerprint(
            task=task_mapping,
            model=model_mapping,
            analysis=components["analysis"],
            checkpoint_epochs=checkpoint_epochs,
        )
        fingerprint_changed = (
            desired_analysis_fingerprint != original_analysis_fingerprint
        )
    else:
        fingerprint_changed = (
            desired_analysis_fingerprint != stored_analysis_fingerprint
        )
    recompute = (
        _diagnostics_require_recompute(data_path, model_config.rank)
        or fingerprint_changed
    )
    logger.info(
        "visualization config source=%s experiment=%s analysis_fingerprint=%s "
        "diagnostics=%s",
        config_source,
        analysis_source_experiment or "-",
        desired_analysis_fingerprint,
        "recompute" if recompute else "reuse",
    )
    if recompute:
        if not checkpoint_paths:
            raise RuntimeError(
                "Analysis config changed or diagnostics are incomplete, but no "
                "epoch-*.pt checkpoints are stored "
                f"in {run_path / 'checkpoints'}"
            )
        loss_config = PhaseNormalizedLossConfig.from_mapping(
            components["train"].get("loss")
        )
        validation_task = create_task(
            task_mapping,
            dt=model_config.dt,
            split="validation",
            loss=loss_config,
        )
        result = analyze_checkpoints(
            checkpoint_paths=checkpoint_paths,
            model_config=model_config,
            task=validation_task,
            config=analysis_config,
            output_dir=diagnostics_dir,
        )
        logger.info(
            "latent dynamics recomputed checkpoints=%d bounds=%s grid_points=%d",
            len(checkpoint_paths),
            _format_bounds(result),
            analysis_config.grid_points,
        )
    selection_result = select_representative_checkpoints(
        metrics_path=metrics_path,
        checkpoint_epochs=checkpoint_epochs,
        config=analysis_config.representative_selection,
        output_path=diagnostics_dir / "checkpoint_selection.yaml",
        source=(
            "visualize:cli-manual"
            if representative_epochs is not None
            else f"visualize:{config_source}:{analysis_config.representative_selection.mode}"
        ),
    )
    representatives = list(selection_result.epochs)
    representative_labels = selection_result.labels
    logger.info(
        "representative checkpoints selected mode=%s transition=%s "
        "terminal_regime=%s epochs=%s",
        analysis_config.representative_selection.mode,
        selection_result.transition_classification,
        selection_result.terminal_regime,
        representatives,
    )
    write_analysis_stage_config(
        stage_config_path,
        config=analysis_mapping,
        fingerprint=desired_analysis_fingerprint,
        checkpoint_epochs=checkpoint_epochs,
        config_source=config_source,
        source_experiment=analysis_source_experiment,
        cli_overrides=analysis_overrides,
    )

    figures_dir = (
        run_path / "figures"
        if output_dir is None
        else Path(output_dir).resolve()
    )
    snapshot_dpi, movie_dpi = resolve_dpi_settings(visualization_mapping)
    should_render_movies = resolve_render_movies(visualization_mapping)
    started_figures = time.perf_counter()
    plot_training_curves(
        metrics_path,
        figures_dir / "loss_and_gradient.png",
        representative_epochs=representatives,
        representative_labels=representative_labels,
        figure_size=visualization_mapping["training_curves_figure_size"],
        loss_y_scale=str(visualization_mapping.get("loss_y_scale", "log")),
        dpi=snapshot_dpi,
    )
    logger.info(
        "figure written file=loss_and_gradient.png elapsed_seconds=%.2fs",
        time.perf_counter() - started_figures,
    )
    started_figures = time.perf_counter()
    plot_representative_outputs(
        data_path,
        figures_dir / "representative_outputs.png",
        representative_epochs=representatives,
        representative_labels=representative_labels,
        dt=model_config.dt,
        panel_width=float(visualization_mapping["outputs_panel_width"]),
        panel_height=float(visualization_mapping["outputs_panel_height"]),
        decision_band_logit_half_width=float(
            visualization_mapping.get("decision_band_logit_half_width", 1.0)
        ),
        response_threshold=response_threshold,
        dpi=snapshot_dpi,
    )
    logger.info(
        "figure written file=representative_outputs.png elapsed_seconds=%.2fs",
        time.perf_counter() - started_figures,
    )
    started_figures = time.perf_counter()
    vector_fields = render_latent_vector_field_collection(
        data_path,
        figures_dir / "latent_vector_field",
        representative_epochs=representatives,
        representative_labels=representative_labels,
        speed_floor=analysis_config.speed_floor,
        figure_size=visualization_mapping["vector_field_figure_size"],
        dpi=snapshot_dpi,
        movie_dpi=movie_dpi,
        render_movie=should_render_movies,
        fps=int(visualization_mapping["movie_fps"]),
        codec=str(visualization_mapping["movie_codec"]),
        arrow_stride=int(visualization_mapping["arrow_stride"]),
        arrow_color=str(visualization_mapping.get("arrow_color", "black")),
        arrow_length_fraction=float(
            visualization_mapping.get("arrow_length_fraction", 0.0075)
        ),
        arrow_width=float(visualization_mapping.get("arrow_width", 0.0012)),
        trajectory_line_width=float(
            visualization_mapping.get("trajectory_line_width", 1.5)
        ),
        decision_band_logit_half_width=float(
            visualization_mapping.get("decision_band_logit_half_width", 1.0)
        ),
        response_threshold=response_threshold,
    )
    if model_config.rank >= 3:
        logger.info(
            "3D vector field omitted rank=%d; using trajectory-neighborhood "
            "diagnostics elapsed_seconds=%.2fs",
            model_config.rank,
            time.perf_counter() - started_figures,
        )
    else:
        logger.info(
            "vector field artifacts written snapshots=%d "
            "movie=%s elapsed_seconds=%.2fs",
            len(vector_fields["snapshots"]),
            "latent_vector_field.mp4" if should_render_movies else "disabled",
            time.perf_counter() - started_figures,
        )
    started_figures = time.perf_counter()
    jacobian_snapshots = render_latent_jacobian_collection(
        data_path,
        figures_dir / "latent_jacobian",
        representative_epochs=representatives,
        representative_labels=representative_labels,
        figure_size=visualization_mapping["vector_field_figure_size"],
        dpi=snapshot_dpi,
        trajectory_line_width=float(
            visualization_mapping.get("trajectory_line_width", 1.5)
        ),
        spectral_abscissa_limit=visualization_mapping.get(
            "jacobian_spectral_abscissa_limit"
        ),
    )
    if model_config.rank >= 3:
        logger.info(
            "2D Jacobian map omitted rank=%d; exact K-D spectra stored in "
            "trajectory-neighborhood diagnostics elapsed_seconds=%.2fs",
            model_config.rank,
            time.perf_counter() - started_figures,
        )
    else:
        logger.info(
            "latent Jacobian artifacts written snapshots=%d elapsed_seconds=%.2fs",
            len(jacobian_snapshots),
            time.perf_counter() - started_figures,
        )
    started_figures = time.perf_counter()
    combined = render_latent_dynamics_collection(
        data_path,
        figures_dir / "latent_dynamics",
        representative_epochs=representatives,
        representative_labels=representative_labels,
        speed_floor=analysis_config.speed_floor,
        figure_size=(
            visualization_mapping["latent_dynamics_3d_figure_size"]
            if model_config.rank >= 3
            else _combined_figure_size(visualization_mapping)
        ),
        dpi=snapshot_dpi,
        movie_dpi=movie_dpi,
        render_movie=should_render_movies,
        fps=int(visualization_mapping["movie_fps"]),
        codec=str(visualization_mapping["movie_codec"]),
        arrow_stride=int(visualization_mapping["arrow_stride"]),
        arrow_color=str(visualization_mapping.get("arrow_color", "black")),
        arrow_length_fraction=float(
            visualization_mapping.get("arrow_length_fraction", 0.0075)
        ),
        arrow_width=float(visualization_mapping.get("arrow_width", 0.0012)),
        trajectory_line_width=float(
            visualization_mapping.get("trajectory_line_width", 1.5)
        ),
        decision_band_logit_half_width=float(
            visualization_mapping.get("decision_band_logit_half_width", 1.0)
        ),
        response_threshold=response_threshold,
        spectral_abscissa_limit=visualization_mapping.get(
            "jacobian_spectral_abscissa_limit"
        ),
        max_trajectories=int(
            visualization_mapping.get("max_3d_trajectories", 64)
        ),
    )
    logger.info(
        "combined latent dynamics artifacts written snapshots=%d "
        "movie=%s elapsed_seconds=%.2fs",
        len(combined["snapshots"]),
        "latent_dynamics.mp4" if should_render_movies else "disabled",
        time.perf_counter() - started_figures,
    )

    desired_visualization_fingerprint = visualization_fingerprint(
        analysis_fingerprint_value=desired_analysis_fingerprint,
        visualization=visualization_mapping,
        representative_epochs=representatives,
    )
    figure_stage_path = run_path / "figures" / "config.yaml"
    write_visualization_stage_config(
        figure_stage_path,
        config=visualization_mapping,
        fingerprint=desired_visualization_fingerprint,
        analysis_fingerprint_value=desired_analysis_fingerprint,
        representative_epochs=representatives,
        config_source=config_source,
        source_experiment=visualization_source_experiment,
        cli_overrides=visualization_overrides,
        output_dir=figures_dir,
    )
    if figures_dir != run_path / "figures":
        write_visualization_stage_config(
            figures_dir / "config.yaml",
            config=visualization_mapping,
            fingerprint=desired_visualization_fingerprint,
            analysis_fingerprint_value=desired_analysis_fingerprint,
            representative_epochs=representatives,
            config_source=config_source,
            source_experiment=visualization_source_experiment,
            cli_overrides=visualization_overrides,
            output_dir=figures_dir,
        )
    logger.info(
        "visualization rerun complete elapsed_seconds=%.1fs figures=%s "
        "visualization_fingerprint=%s",
        time.perf_counter() - started,
        figures_dir,
        desired_visualization_fingerprint,
    )
    return figures_dir


def _diagnostics_require_recompute(data_path: Path, rank: int = 2) -> bool:
    """Upgrade diagnostics when dimension-specific structured fields are absent."""
    if not data_path.is_file():
        return True
    with np.load(data_path) as data:
        if rank >= 3:
            return not {
                "coordinate_dimension",
                "coordinate_kind",
                "trajectory_latent",
                "neighborhood_latent",
                "neighborhood_q",
                "neighborhood_jacobian_eigenvalues",
                "neighborhood_spectral_abscissa",
                "neighborhood_low_q_mask",
                "neighborhood_near_zero_mask",
            }.issubset(data.files)
        return not {
            "jacobian_eigenvalues",
            "jacobian_spectral_abscissa",
            "speed_minimum_coordinates",
            "speed_minimum_classification",
            "trial_s1_step",
            "trial_s2_step",
            "trial_go_step",
            "trial_response_step",
            "trial_steps",
            "square_coordinate_bounds",
        }.issubset(data.files)


def _resolve_stage_mappings(
    resolved: Mapping[str, Any],
    run_path: Path,
    *,
    config_source: str,
    experiment: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any], str | None, str | None]:
    """Resolve post-training components without changing frozen run inputs."""
    if config_source not in {"current", "original", "last"}:
        raise ConfigError("config_source must be one of: current, original, last")
    if experiment is not None and config_source != "current":
        raise ConfigError(
            "--experiment can only be used with --config-source current"
        )

    components = resolved["components"]
    if config_source == "original":
        source = resolved.get("source")
        return (
            deepcopy(components["analysis"]),
            deepcopy(components["visualization"]),
            str(source) if source is not None else None,
            str(source) if source is not None else None,
        )

    if config_source == "last":
        analysis, analysis_record = load_stage_component(
            run_path / "diagnostics" / "config.yaml",
            expected_kind="analysis",
        )
        visualization, visualization_record = load_stage_component(
            run_path / "figures" / "config.yaml",
            expected_kind="visualization",
        )
        return (
            analysis,
            visualization,
            _optional_string(analysis_record.get("source_experiment")),
            _optional_string(visualization_record.get("source_experiment")),
        )

    recipe_path = experiment if experiment is not None else resolved.get("source")
    if recipe_path is None:
        raise ConfigError(
            "Run config has no source experiment; pass --experiment or use "
            "--config-source original/last"
        )
    current = resolve_experiment(recipe_path)
    missing = {"analysis", "visualization"} - set(current.components)
    if missing:
        raise ConfigError(
            "Current experiment is missing post-training components: "
            + ", ".join(sorted(missing))
        )
    return (
        deepcopy(current.components["analysis"]),
        deepcopy(current.components["visualization"]),
        current.source,
        current.source,
    )


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _stored_stage_fingerprint(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = load_yaml(path).get("fingerprint")
    return value if isinstance(value, str) and value else None


def _stored_latent_analysis_fingerprint(
    path: Path,
    *,
    task: Mapping[str, Any],
    model: Mapping[str, Any],
) -> str | None:
    """Canonicalize old stage records with the current fingerprint boundary."""
    if not path.is_file():
        return None
    record = load_yaml(path)
    config = record.get("config")
    checkpoint_epochs = record.get("checkpoint_epochs")
    if not isinstance(config, Mapping) or not isinstance(
        checkpoint_epochs, Sequence
    ):
        return _stored_stage_fingerprint(path)
    return analysis_fingerprint(
        task=task,
        model=model,
        analysis=config,
        checkpoint_epochs=[int(epoch) for epoch in checkpoint_epochs],
    )


def _stored_diagnostic_epochs(data_path: Path) -> list[int]:
    if not data_path.is_file():
        return []
    with np.load(data_path) as data:
        if "epochs" not in data.files:
            return []
        return [int(epoch) for epoch in data["epochs"]]


def _combined_figure_size(config: Mapping[str, Any]) -> list[float]:
    configured = config.get("latent_dynamics_figure_size")
    if configured is not None:
        return [float(value) for value in configured]
    single = config["vector_field_figure_size"]
    return [2.0 * float(single[0]), float(single[1])]


def _epoch_checkpoint_paths(run_dir: Path) -> dict[int, Path]:
    """Map epoch numbers to the periodic ``epoch-*.pt`` checkpoints."""
    paths: dict[int, Path] = {}
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("epoch-*.pt"):
            match = _EPOCH_CHECKPOINT.fullmatch(path.name)
            if match:
                paths[int(match.group(1))] = path
    return dict(sorted(paths.items()))


def _requested_named_checkpoint_paths(
    run_dir: Path,
    requested_epochs: Sequence[int],
    periodic_paths: Mapping[int, Path],
) -> dict[int, Path]:
    """Resolve manual requests that correspond to initial/best/final files."""
    missing = set(int(epoch) for epoch in requested_epochs) - set(periodic_paths)
    if not missing:
        return {}
    resolved: dict[int, Path] = {}
    for name in ("initial.pt", "best.pt", "final.pt"):
        path = run_dir / "checkpoints" / name
        if not path.is_file():
            continue
        epoch = int(load_checkpoint(path)["epoch"])
        if epoch in missing:
            resolved[epoch] = path
    return resolved


def _format_bounds(result: Any) -> str:
    """Short bounds display for the log line."""
    with np.load(result.data_path) as data:
        bounds = (
            data["bounds"].reshape(-1)
            if "bounds" in data.files
            else data["display_bounds"].reshape(-1)
        )
    return "[" + ", ".join(f"{value:.3g}" for value in bounds) + "]"
