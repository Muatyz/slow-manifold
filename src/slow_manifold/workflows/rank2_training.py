"""End-to-end low-rank training, diagnosis, and figure workflow."""

from __future__ import annotations

import csv
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from slow_manifold.analysis import (
    AnalysisConfig,
    analyze_checkpoints,
    select_representative_checkpoints,
)
from slow_manifold.config import (
    ResolvedExperiment,
    dump_yaml,
    load_yaml,
    resolve_experiment,
)
from slow_manifold.models import Rank2CTRNN, Rank2CTRNNConfig
from slow_manifold.tasks import create_task
from slow_manifold.training import TrainConfig, train_model
from slow_manifold.training.checkpoint import load_checkpoint
from slow_manifold.utils import (
    collect_runtime_metadata,
    derive_seed,
    get_logger,
    make_rng,
    setup_run_logging,
    startup_summary_lines,
)
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
    visualization_fingerprint,
    write_analysis_stage_config,
    write_visualization_stage_config,
)


def run_rank2_training(
    experiment_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> Path:
    """Run a new traceable training experiment and generate requested figures.

    Maintains the run lifecycle in ``status.yaml`` (``running`` ->
    ``complete``/``failed``/``interrupted``) and appends every event to
    ``run.log`` inside the run directory. The run identity is derived from
    the resolved training configuration and seed.
    """
    experiment = _with_cli_overrides(
        resolve_experiment(experiment_path),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    run_dir = (
        experiment.default_run_dir()
        if output_dir is None
        else Path(output_dir).resolve()
    )
    if run_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {run_dir}"
        )
    run_dir.mkdir(parents=True)

    setup_run_logging(run_dir)
    logger = get_logger("run")
    started_utc = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    logger.info(
        "run started experiment=%s condition=%s fingerprint=%s seed=%d "
        "device=%s output=%s",
        experiment.name,
        experiment.condition_label or "-",
        experiment.condition_fingerprint,
        experiment.seed,
        experiment.components.get("train", {}).get("device", "unknown"),
        run_dir,
    )
    _write_status(run_dir, status="running", started_at_utc=started_utc)
    try:
        status_extra = _run_experiment_pipeline(experiment, run_dir)
    except KeyboardInterrupt:
        logger.error("run interrupted (Ctrl+C)")
        logger.exception("interrupted traceback")
        _write_status(
            run_dir,
            status="interrupted",
            started_at_utc=started_utc,
            wall_started=wall_started,
            last_step=_last_step_from_metrics(run_dir),
            error={
                "type": "KeyboardInterrupt",
                "message": "Interrupted by Ctrl+C",
            },
        )
        raise
    except Exception as error:
        logger.exception("run failed: %s: %s", type(error).__name__, error)
        _write_status(
            run_dir,
            status="failed",
            started_at_utc=started_utc,
            wall_started=wall_started,
            last_step=_last_step_from_metrics(run_dir),
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    elapsed_seconds = time.perf_counter() - wall_started
    _write_status(
        run_dir,
        status="complete",
        started_at_utc=started_utc,
        wall_started=wall_started,
        extra=status_extra,
    )
    logger.info("run complete elapsed_seconds=%.1fs", elapsed_seconds)
    return run_dir


def resume_rank2_training(run_dir: str | Path) -> Path:
    """Continue an interrupted/failed run from its latest consistent checkpoint."""
    run_path = Path(run_dir).resolve()
    experiment = _load_frozen_experiment(run_path)
    status_path = run_path / "status.yaml"
    previous_status = load_yaml(status_path)
    if previous_status.get("status") == "complete":
        raise ValueError(f"Run is already complete and cannot be resumed: {run_path}")
    checkpoint_path, resume_payload = _select_resume_checkpoint(
        run_path, experiment
    )

    setup_run_logging(run_path)
    logger = get_logger("run")
    started_utc = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    resume_count = int(previous_status.get("resume_count", 0)) + 1
    resume_details = {
        "resume_count": resume_count,
        "resumed_from_epoch": int(resume_payload["epoch"]),
        "resume_checkpoint": str(checkpoint_path),
        "previous_status": str(previous_status.get("status", "unknown")),
        "initial_started_at_utc": previous_status.get(
            "initial_started_at_utc", previous_status.get("started_at_utc")
        ),
    }
    logger.info(
        "run resume started checkpoint=%s from_epoch=%d target_epoch=%d "
        "previous_status=%s resume_count=%d",
        checkpoint_path,
        int(resume_payload["epoch"]),
        int(experiment.components["train"]["epochs"]),
        resume_details["previous_status"],
        resume_count,
    )
    _write_status(
        run_path,
        status="running",
        started_at_utc=started_utc,
        extra=resume_details,
    )
    try:
        status_extra = _run_experiment_pipeline(
            experiment, run_path, resume_payload=resume_payload
        )
    except KeyboardInterrupt:
        logger.error("resumed run interrupted (Ctrl+C)")
        logger.exception("interrupted traceback")
        _write_status(
            run_path,
            status="interrupted",
            started_at_utc=started_utc,
            wall_started=wall_started,
            last_step=_last_step_from_metrics(run_path),
            error={
                "type": "KeyboardInterrupt",
                "message": "Interrupted by Ctrl+C",
            },
            extra=resume_details,
        )
        raise
    except Exception as error:
        logger.exception("resumed run failed: %s: %s", type(error).__name__, error)
        _write_status(
            run_path,
            status="failed",
            started_at_utc=started_utc,
            wall_started=wall_started,
            last_step=_last_step_from_metrics(run_path),
            error={"type": type(error).__name__, "message": str(error)},
            extra=resume_details,
        )
        raise
    _write_status(
        run_path,
        status="complete",
        started_at_utc=started_utc,
        wall_started=wall_started,
        extra={**resume_details, **status_extra},
    )
    logger.info(
        "run resume complete elapsed_seconds=%.1fs", time.perf_counter() - wall_started
    )
    return run_path


def _run_experiment_pipeline(
    experiment: ResolvedExperiment,
    run_dir: Path,
    *,
    resume_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve components, train or resume, then diagnose and plot."""
    logger = get_logger("run")
    _require_components(
        experiment,
        "task",
        "model",
        "train",
        "analysis",
        "visualization",
        "reporting",
    )
    model_config = Rank2CTRNNConfig.from_mapping(experiment.components["model"])
    train_config = TrainConfig.from_mapping(experiment.components["train"])
    train_task = create_task(
        experiment.components["task"],
        dt=model_config.dt,
        split="train",
        loss=train_config.loss,
    )
    validation_task = create_task(
        experiment.components["task"],
        dt=model_config.dt,
        split="validation",
        loss=train_config.loss,
    )
    analysis_config = AnalysisConfig.from_mapping(experiment.components["analysis"])
    visualization_config = experiment.components["visualization"]
    snapshot_dpi, movie_dpi = resolve_dpi_settings(visualization_config)
    render_movies = resolve_render_movies(visualization_config)
    reporting_config = experiment.components["reporting"]
    response_threshold = float(
        experiment.components["task"].get("evaluation", {}).get(
            "response_threshold", 0.5
        )
    )
    _validate_cross_component_config(
        model_config, analysis_config, train_config, len(train_task.input_names)
    )

    for line in startup_summary_lines(experiment.components, reporting_config):
        logger.info("configuration %s", line)

    resolved = experiment.as_dict()
    resolved["run"] = {
        "kind": "low_rank_training",
        "condition_label": experiment.condition_label,
        "condition_fingerprint": experiment.condition_fingerprint,
        "output_dir": str(run_dir),
    }
    if resume_payload is None:
        dump_yaml(resolved, run_dir / "config.yaml")
    rng_streams = {
        "model": "model:initialization",
        "task_train": "task:train",
        "task_validation": "task:validation",
        "initial_state_train": "rollout:initial_state:train",
        "neural_noise_train": "rollout:neural_noise:train",
        "initial_state_validation": "rollout:initial_state:validation",
        "neural_noise_validation": "rollout:neural_noise:validation",
    }
    if resume_payload is None:
        dump_yaml(
            collect_runtime_metadata(
                experiment_name=experiment.name,
                seed=experiment.seed,
                rng_streams=rng_streams,
                packages=(
                    "numpy",
                    "matplotlib",
                    "pyyaml",
                    "torch",
                    "imageio-ffmpeg",
                ),
                extra={
                    "device": train_config.device,
                    "condition_label": experiment.condition_label,
                    "condition_fingerprint": experiment.condition_fingerprint,
                    "num_threads": train_config.num_threads,
                    "interop_threads": torch.get_num_interop_threads(),
                    **_device_metadata(train_config.device),
                    "dtype": model_config.dtype,
                    "gradient_norm_definition": (
                        "||grad_M L||_F + ||grad_N L||_F, measured before clipping"
                    ),
                    "neural_noise_injection": "activation input, inside tanh",
                    "analysis_rollout": "zero initial state, no neural noise",
                },
            ),
            run_dir / "metadata.yaml",
        )

    torch_seed = derive_seed(experiment.seed, rng_streams["model"])
    torch.manual_seed(torch_seed)
    model_generator = torch.Generator(device="cpu").manual_seed(torch_seed)
    model = Rank2CTRNN(model_config, generator=model_generator)
    train_rng = make_rng(experiment.seed, rng_streams["task_train"])
    validation_rng = make_rng(experiment.seed, rng_streams["task_validation"])
    rollout_seeds = {
        name: derive_seed(experiment.seed, namespace)
        for name, namespace in rng_streams.items()
        if name.startswith("initial_state_") or name.startswith("neural_noise_")
    }
    validation_batch = validation_task.generate_batch(
        train_config.validation_batch_size, validation_rng
    )
    if resume_payload is None:
        _save_validation_batch(validation_batch, run_dir / "diagnostics")

    selection_config = analysis_config.representative_selection
    snapshot_epochs = (
        [
            epoch
            for epoch in selection_config.manual_epochs
            if epoch <= train_config.epochs
        ]
        if selection_config.mode == "manual"
        else []
    )
    training_result = train_model(
        model=model,
        train_task=train_task,
        validation_batch=validation_batch,
        train_rng=train_rng,
        rollout_seeds=rollout_seeds,
        config=train_config,
        run_dir=run_dir,
        snapshot_epochs=snapshot_epochs,
        resolved_config=resolved,
        resume_payload=resume_payload,
    )

    selection_result = select_representative_checkpoints(
        metrics_path=run_dir / "metrics.csv",
        checkpoint_epochs=training_result.checkpoint_paths,
        config=selection_config,
        output_path=run_dir / "diagnostics" / "checkpoint_selection.yaml",
        source=f"training:{selection_config.mode}",
    )
    logger.info(
        "representative checkpoints selected mode=%s transition=%s "
        "terminal_regime=%s epochs=%s",
        selection_config.mode,
        selection_result.transition_classification,
        selection_result.terminal_regime,
        list(selection_result.epochs),
    )

    analysis_logger = get_logger("analysis")
    analysis_started = time.perf_counter()
    latent_result = analyze_checkpoints(
        checkpoint_paths=training_result.checkpoint_paths,
        model_config=model_config,
        task=validation_task,
        config=analysis_config,
        output_dir=run_dir / "diagnostics",
        representative_epochs=selection_result.epochs,
    )
    checkpoint_epochs = list(training_result.checkpoint_paths)
    resolved_analysis_fingerprint = analysis_fingerprint(
        task=experiment.components["task"],
        model=experiment.components["model"],
        analysis=experiment.components["analysis"],
        checkpoint_epochs=checkpoint_epochs,
        slow_point_epochs=selection_result.epochs,
    )
    write_analysis_stage_config(
        run_dir / "diagnostics" / "config.yaml",
        config=experiment.components["analysis"],
        fingerprint=resolved_analysis_fingerprint,
        checkpoint_epochs=checkpoint_epochs,
        config_source="training",
        source_experiment=experiment.source,
        slow_point_epochs=selection_result.epochs,
    )
    analysis_logger.info(
        "latent analysis complete checkpoints=%d elapsed_seconds=%.2fs",
        len(training_result.checkpoint_paths),
        time.perf_counter() - analysis_started,
    )

    visualization_logger = get_logger("visualization")
    figures = run_dir / "figures"
    started = time.perf_counter()
    plot_training_curves(
        run_dir / "metrics.csv",
        figures / "loss_and_gradient.png",
        representative_epochs=selection_result.epochs,
        representative_labels=selection_result.labels,
        figure_size=visualization_config["training_curves_figure_size"],
        loss_y_scale=str(visualization_config.get("loss_y_scale", "log")),
        dpi=snapshot_dpi,
    )
    visualization_logger.info(
        "figure written file=loss_and_gradient.png elapsed_seconds=%.2fs",
        time.perf_counter() - started,
    )
    started = time.perf_counter()
    plot_representative_outputs(
        latent_result.data_path,
        figures / "representative_outputs.png",
        representative_epochs=selection_result.epochs,
        representative_labels=selection_result.labels,
        dt=model_config.dt,
        panel_width=float(visualization_config["outputs_panel_width"]),
        panel_height=float(visualization_config["outputs_panel_height"]),
        decision_band_logit_half_width=float(
            visualization_config.get("decision_band_logit_half_width", 1.0)
        ),
        response_threshold=response_threshold,
        dpi=snapshot_dpi,
    )
    visualization_logger.info(
        "figure written file=representative_outputs.png elapsed_seconds=%.2fs",
        time.perf_counter() - started,
    )
    started = time.perf_counter()
    vector_fields = render_latent_vector_field_collection(
        latent_result.data_path,
        figures / "latent_vector_field",
        representative_epochs=selection_result.epochs,
        representative_labels=selection_result.labels,
        speed_floor=analysis_config.speed_floor,
        figure_size=visualization_config["vector_field_figure_size"],
        dpi=snapshot_dpi,
        movie_dpi=movie_dpi,
        render_movie=render_movies,
        fps=int(visualization_config["movie_fps"]),
        codec=str(visualization_config["movie_codec"]),
        arrow_stride=int(visualization_config["arrow_stride"]),
        arrow_color=str(visualization_config.get("arrow_color", "black")),
        arrow_length_fraction=float(
            visualization_config.get("arrow_length_fraction", 0.0075)
        ),
        arrow_width=float(visualization_config.get("arrow_width", 0.0012)),
        trajectory_line_width=float(
            visualization_config.get("trajectory_line_width", 1.5)
        ),
        decision_band_logit_half_width=float(
            visualization_config.get("decision_band_logit_half_width", 1.0)
        ),
        response_threshold=response_threshold,
    )
    if model_config.rank >= 3:
        visualization_logger.info(
            "3D vector field omitted rank=%d; using trajectory-neighborhood "
            "diagnostics elapsed_seconds=%.2fs",
            model_config.rank,
            time.perf_counter() - started,
        )
    else:
        visualization_logger.info(
            "vector field artifacts written snapshots=%d "
            "movie=%s elapsed_seconds=%.2fs",
            len(vector_fields["snapshots"]),
            "latent_vector_field.mp4" if render_movies else "disabled",
            time.perf_counter() - started,
        )
    started = time.perf_counter()
    jacobian_snapshots = render_latent_jacobian_collection(
        latent_result.data_path,
        figures / "latent_jacobian",
        representative_epochs=selection_result.epochs,
        representative_labels=selection_result.labels,
        figure_size=visualization_config["vector_field_figure_size"],
        dpi=snapshot_dpi,
        trajectory_line_width=float(
            visualization_config.get("trajectory_line_width", 1.5)
        ),
        spectral_abscissa_limit=visualization_config.get(
            "jacobian_spectral_abscissa_limit"
        ),
    )
    if model_config.rank >= 3:
        visualization_logger.info(
            "2D Jacobian map omitted rank=%d; exact K-D spectra stored in "
            "trajectory-neighborhood diagnostics elapsed_seconds=%.2fs",
            model_config.rank,
            time.perf_counter() - started,
        )
    else:
        visualization_logger.info(
            "latent Jacobian artifacts written snapshots=%d elapsed_seconds=%.2fs",
            len(jacobian_snapshots),
            time.perf_counter() - started,
        )
    started = time.perf_counter()
    combined = render_latent_dynamics_collection(
        latent_result.data_path,
        figures / "latent_dynamics",
        representative_epochs=selection_result.epochs,
        representative_labels=selection_result.labels,
        speed_floor=analysis_config.speed_floor,
        figure_size=(
            visualization_config["latent_dynamics_3d_figure_size"]
            if model_config.rank >= 3
            else _combined_figure_size(visualization_config)
        ),
        dpi=snapshot_dpi,
        movie_dpi=movie_dpi,
        render_movie=render_movies,
        fps=int(visualization_config["movie_fps"]),
        codec=str(visualization_config["movie_codec"]),
        arrow_stride=int(visualization_config["arrow_stride"]),
        arrow_color=str(visualization_config.get("arrow_color", "black")),
        arrow_length_fraction=float(
            visualization_config.get("arrow_length_fraction", 0.0075)
        ),
        arrow_width=float(visualization_config.get("arrow_width", 0.0012)),
        trajectory_line_width=float(
            visualization_config.get("trajectory_line_width", 1.5)
        ),
        decision_band_logit_half_width=float(
            visualization_config.get("decision_band_logit_half_width", 1.0)
        ),
        response_threshold=response_threshold,
        spectral_abscissa_limit=visualization_config.get(
            "jacobian_spectral_abscissa_limit"
        ),
        max_trajectories=int(
            visualization_config.get("max_3d_trajectories", 64)
        ),
        trajectory_3d_alpha=float(
            visualization_config.get("trajectory_3d_alpha", 0.62)
        ),
        trajectory_3d_cue_alpha_scale=float(
            visualization_config.get("trajectory_3d_cue_alpha_scale", 0.55)
        ),
        trajectory_3d_cue_line_width_scale=float(
            visualization_config.get(
                "trajectory_3d_cue_line_width_scale", 0.8
            )
        ),
        trajectory_3d_view_elevation=float(
            visualization_config.get("trajectory_3d_view_elevation", 26.0)
        ),
        trajectory_3d_view_azimuth=float(
            visualization_config.get("trajectory_3d_view_azimuth", -68.0)
        ),
        single_trajectory_enabled=visualization_config.get(
            "single_trajectory_enabled", True
        ),
        single_trajectory_figure_size=visualization_config.get(
            "single_trajectory_figure_size", [7.0, 6.0]
        ),
        single_trajectory_interval_quantile=float(
            visualization_config.get(
                "single_trajectory_interval_quantile", 0.5
            )
        ),
    )
    visualization_logger.info(
        "combined latent dynamics artifacts written snapshots=%d "
        "movie=%s elapsed_seconds=%.2fs",
        len(combined["snapshots"]),
        "latent_dynamics.mp4" if render_movies else "disabled",
        time.perf_counter() - started,
    )
    resolved_visualization_fingerprint = visualization_fingerprint(
        analysis_fingerprint_value=resolved_analysis_fingerprint,
        visualization=visualization_config,
        representative_epochs=selection_result.epochs,
    )
    write_visualization_stage_config(
        figures / "config.yaml",
        config=visualization_config,
        fingerprint=resolved_visualization_fingerprint,
        analysis_fingerprint_value=resolved_analysis_fingerprint,
        representative_epochs=selection_result.epochs,
        config_source="training",
        source_experiment=experiment.source,
        output_dir=figures,
    )
    return {
        "best_epoch": training_result.best_epoch,
        "final_epoch": train_config.epochs,
        "representative_epochs": list(selection_result.epochs),
        "representative_transition": selection_result.transition_classification,
        "terminal_regime": selection_result.terminal_regime,
    }


def _write_status(
    run_dir: Path,
    *,
    status: str,
    started_at_utc: datetime | None = None,
    wall_started: float | None = None,
    last_step: int | None = None,
    error: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write the run lifecycle entry to ``status.yaml``."""
    data: dict[str, Any] = {"status": status}
    if started_at_utc is not None:
        data["started_at_utc"] = started_at_utc.isoformat()
    if wall_started is not None:
        data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        data["elapsed_seconds"] = round(time.perf_counter() - wall_started, 1)
    if last_step is not None:
        data["last_step"] = last_step
    if error is not None:
        data["error"] = dict(error)
    if extra:
        data.update(extra)
    dump_yaml(data, run_dir / "status.yaml")


def _combined_figure_size(config: Mapping[str, Any]) -> list[float]:
    configured = config.get("latent_dynamics_figure_size")
    if configured is not None:
        return [float(value) for value in configured]
    single = config["vector_field_figure_size"]
    return [2.0 * float(single[0]), float(single[1])]


def _last_step_from_metrics(run_dir: Path) -> int | None:
    """Recover the last completed epoch from ``metrics.csv`` after a failure."""
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    return int(rows[-1]["epoch"])


def _load_frozen_experiment(run_dir: Path) -> ResolvedExperiment:
    """Reconstruct the authoritative experiment from an immutable run config."""
    config_path = run_dir / "config.yaml"
    frozen = load_yaml(config_path)
    components = frozen.get("components")
    if not isinstance(components, Mapping):
        raise ValueError(f"Run config has no resolved components: {config_path}")
    identity = frozen.get("identity", {})
    label_fields = (
        identity.get("label_fields", {}) if isinstance(identity, Mapping) else {}
    )
    if not isinstance(label_fields, Mapping):
        raise ValueError("Run config identity.label_fields must be a mapping")
    component_sources = frozen.get("component_sources", {})
    if not isinstance(component_sources, Mapping):
        raise ValueError("Run config component_sources must be a mapping")
    experiment = ResolvedExperiment(
        name=str(frozen["name"]),
        seed=int(frozen["seed"]),
        components=deepcopy(dict(components)),
        component_sources={
            str(name): str(source) for name, source in component_sources.items()
        },
        source=str(frozen.get("source", config_path)),
        identity_fields={
            str(name): str(path) for name, path in label_fields.items()
        },
    )
    run_record = frozen.get("run", {})
    if not isinstance(run_record, Mapping):
        raise ValueError("Run config run record must be a mapping")
    stored_fingerprint = run_record.get("condition_fingerprint")
    if stored_fingerprint != experiment.condition_fingerprint:
        raise ValueError(
            "Frozen task/model/train fingerprint does not match the run record"
        )
    return experiment


def _select_resume_checkpoint(
    run_dir: Path, experiment: ResolvedExperiment
) -> tuple[Path, dict[str, Any]]:
    """Choose the newest checkpoint consistent with the durable metric trace."""
    last_metric_epoch = _last_step_from_metrics(run_dir)
    if last_metric_epoch is None:
        raise ValueError("Cannot resume a run without completed metric rows")
    target_epoch = int(experiment.components["train"]["epochs"])
    checkpoint_dir = run_dir / "checkpoints"
    candidate_paths = list(checkpoint_dir.glob("epoch-*.pt"))
    candidate_paths.extend(
        path
        for path in (
            checkpoint_dir / "initial.pt",
            checkpoint_dir / "best.pt",
            checkpoint_dir / "final.pt",
        )
        if path.is_file()
    )
    candidates: list[tuple[int, int, Path, dict[str, Any]]] = []
    priority = {"initial.pt": 0, "best.pt": 1, "final.pt": 3}
    for path in candidate_paths:
        payload = load_checkpoint(path)
        epoch = int(payload.get("epoch", -1))
        if epoch < 0 or epoch > last_metric_epoch or epoch > target_epoch:
            continue
        _validate_resume_checkpoint(payload, experiment)
        candidates.append((epoch, priority.get(path.name, 2), path, payload))
    if not candidates:
        raise ValueError(
            "No checkpoint is consistent with metrics.csv and the frozen config"
        )
    _, _, path, payload = max(candidates, key=lambda item: (item[0], item[1]))
    return path, payload


def _validate_resume_checkpoint(
    payload: Mapping[str, Any], experiment: ResolvedExperiment
) -> None:
    required = {"model_state", "optimizer_state", "rng_state", "resolved_config"}
    missing = required - set(payload)
    if missing:
        raise ValueError(
            "Checkpoint cannot resume training; missing "
            + ", ".join(sorted(missing))
        )
    stored = payload["resolved_config"]
    if not isinstance(stored, Mapping):
        raise ValueError("Checkpoint resolved_config must be a mapping")
    stored_components = stored.get("components")
    if not isinstance(stored_components, Mapping):
        raise ValueError("Checkpoint has no resolved components")
    for name in ("task", "model", "train"):
        if stored_components.get(name) != experiment.components.get(name):
            raise ValueError(
                f"Checkpoint {name} config does not match the frozen run config"
            )


def _with_cli_overrides(
    experiment: ResolvedExperiment,
    *,
    epochs: int | None,
    batch_size: int | None,
    device: str | None,
) -> ResolvedExperiment:
    components = deepcopy(experiment.components)
    train = components.get("train")
    if isinstance(train, dict):
        if epochs is not None:
            train["epochs"] = epochs
        if batch_size is not None:
            train["batch_size"] = batch_size
        if device is not None:
            train["device"] = device
    return ResolvedExperiment(
        name=experiment.name,
        seed=experiment.seed,
        components=components,
        component_sources=deepcopy(experiment.component_sources),
        source=experiment.source,
        identity_fields=deepcopy(experiment.identity_fields),
    )


def _require_components(experiment: ResolvedExperiment, *names: str) -> None:
    missing = set(names) - set(experiment.components)
    if missing:
        raise ValueError(f"Missing experiment components: {', '.join(sorted(missing))}")


def _validate_cross_component_config(
    model: Rank2CTRNNConfig,
    analysis: AnalysisConfig,
    train: TrainConfig,
    task_input_size: int,
) -> None:
    if model.input_size != task_input_size or model.output_size != 1:
        raise ValueError(
            f"Task requires input_size={task_input_size} and output_size=1"
        )
    if len(analysis.input_condition) != model.input_size:
        raise ValueError("Analysis input_condition must match model input_size")
    manual_epochs = analysis.representative_selection.manual_epochs
    if any(epoch > train.epochs for epoch in manual_epochs):
        raise ValueError(
            "Manual representative epochs must not exceed the requested "
            "training duration"
        )


def _save_validation_batch(batch, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "validation_batch.npz",
        inputs=batch.inputs,
        target=batch.target,
        loss_mask=batch.loss_mask,
        valid_mask=batch.valid_mask,
    )
    dump_yaml(
        {"trials": [asdict(item) for item in batch.metadata]},
        output_dir / "validation_trials.yaml",
    )


def _device_metadata(device: str) -> dict[str, object]:
    if device != "cuda":
        return {}
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": properties.name,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory_gib": round(properties.total_memory / 2**30, 2),
    }
