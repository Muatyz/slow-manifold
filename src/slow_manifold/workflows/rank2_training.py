"""End-to-end rank-2 training, diagnosis, and figure workflow."""

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

from slow_manifold.analysis import AnalysisConfig, analyze_checkpoints
from slow_manifold.config import ResolvedExperiment, dump_yaml, resolve_experiment
from slow_manifold.models import Rank2CTRNN, Rank2CTRNNConfig
from slow_manifold.tasks import create_task
from slow_manifold.training import TrainConfig, train_model
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


def _run_experiment_pipeline(
    experiment: ResolvedExperiment, run_dir: Path
) -> dict[str, Any]:
    """Resolve components, train, diagnose, and plot for a fresh run."""
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
    train_task = create_task(
        experiment.components["task"], dt=model_config.dt, split="train"
    )
    validation_task = create_task(
        experiment.components["task"], dt=model_config.dt, split="validation"
    )
    train_config = TrainConfig.from_mapping(experiment.components["train"])
    analysis_config = AnalysisConfig.from_mapping(experiment.components["analysis"])
    visualization_config = experiment.components["visualization"]
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
        "kind": "rank2_training",
        "condition_label": experiment.condition_label,
        "condition_fingerprint": experiment.condition_fingerprint,
        "output_dir": str(run_dir),
    }
    dump_yaml(resolved, run_dir / "config.yaml")
    rng_streams = {
        "model": "model:initialization",
        "task_train": "task:train",
        "task_validation": "task:validation",
    }
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
    validation_batch = validation_task.generate_batch(
        train_config.validation_batch_size, validation_rng
    )
    _save_validation_batch(validation_batch, run_dir / "diagnostics")

    snapshot_epochs = [
        epoch
        for epoch in analysis_config.representative_epochs
        if epoch <= train_config.epochs
    ]
    training_result = train_model(
        model=model,
        train_task=train_task,
        validation_batch=validation_batch,
        train_rng=train_rng,
        config=train_config,
        run_dir=run_dir,
        snapshot_epochs=snapshot_epochs,
        resolved_config=resolved,
    )

    analysis_logger = get_logger("analysis")
    analysis_started = time.perf_counter()
    latent_result = analyze_checkpoints(
        checkpoint_paths=training_result.checkpoint_paths,
        model_config=model_config,
        task=validation_task,
        config=analysis_config,
        output_dir=run_dir / "diagnostics",
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
        representative_epochs=latent_result.representative_epochs,
        figure_size=visualization_config["training_curves_figure_size"],
    )
    visualization_logger.info(
        "figure written file=loss_and_gradient.png elapsed_seconds=%.2fs",
        time.perf_counter() - started,
    )
    started = time.perf_counter()
    plot_representative_outputs(
        latent_result.data_path,
        figures / "representative_outputs.png",
        representative_epochs=latent_result.representative_epochs,
        dt=model_config.dt,
        panel_width=float(visualization_config["outputs_panel_width"]),
        panel_height=float(visualization_config["outputs_panel_height"]),
        decision_band_logit_half_width=float(
            visualization_config.get("decision_band_logit_half_width", 1.0)
        ),
        response_threshold=response_threshold,
    )
    visualization_logger.info(
        "figure written file=representative_outputs.png elapsed_seconds=%.2fs",
        time.perf_counter() - started,
    )
    started = time.perf_counter()
    vector_fields = render_latent_vector_field_collection(
        latent_result.data_path,
        figures / "latent_vector_field",
        representative_epochs=latent_result.representative_epochs,
        speed_floor=analysis_config.speed_floor,
        figure_size=visualization_config["vector_field_figure_size"],
        dpi=int(visualization_config["vector_field_dpi"]),
        fps=int(visualization_config["movie_fps"]),
        codec=str(visualization_config["movie_codec"]),
        arrow_stride=int(visualization_config["arrow_stride"]),
        arrow_color=str(visualization_config.get("arrow_color", "black")),
        arrow_length_fraction=float(
            visualization_config.get("arrow_length_fraction", 0.012)
        ),
        arrow_width=float(visualization_config.get("arrow_width", 0.002)),
        decision_band_logit_half_width=float(
            visualization_config.get("decision_band_logit_half_width", 1.0)
        ),
        response_threshold=response_threshold,
    )
    visualization_logger.info(
        "vector field artifacts written snapshots=%d movie=latent_vector_field.mp4 "
        "elapsed_seconds=%.2fs",
        len(vector_fields["snapshots"]),
        time.perf_counter() - started,
    )
    started = time.perf_counter()
    jacobian_snapshots = render_latent_jacobian_collection(
        latent_result.data_path,
        figures / "latent_jacobian",
        representative_epochs=latent_result.representative_epochs,
        figure_size=visualization_config["vector_field_figure_size"],
        dpi=int(visualization_config["vector_field_dpi"]),
        spectral_abscissa_limit=visualization_config.get(
            "jacobian_spectral_abscissa_limit"
        ),
    )
    visualization_logger.info(
        "latent Jacobian artifacts written snapshots=%d elapsed_seconds=%.2fs",
        len(jacobian_snapshots),
        time.perf_counter() - started,
    )
    started = time.perf_counter()
    combined = render_latent_dynamics_collection(
        latent_result.data_path,
        figures / "latent_dynamics",
        representative_epochs=latent_result.representative_epochs,
        speed_floor=analysis_config.speed_floor,
        figure_size=_combined_figure_size(visualization_config),
        dpi=int(visualization_config["vector_field_dpi"]),
        fps=int(visualization_config["movie_fps"]),
        codec=str(visualization_config["movie_codec"]),
        arrow_stride=int(visualization_config["arrow_stride"]),
        arrow_color=str(visualization_config.get("arrow_color", "black")),
        arrow_length_fraction=float(
            visualization_config.get("arrow_length_fraction", 0.012)
        ),
        arrow_width=float(visualization_config.get("arrow_width", 0.002)),
        decision_band_logit_half_width=float(
            visualization_config.get("decision_band_logit_half_width", 1.0)
        ),
        response_threshold=response_threshold,
        spectral_abscissa_limit=visualization_config.get(
            "jacobian_spectral_abscissa_limit"
        ),
    )
    visualization_logger.info(
        "combined latent dynamics artifacts written snapshots=%d "
        "movie=latent_dynamics.mp4 elapsed_seconds=%.2fs",
        len(combined["snapshots"]),
        time.perf_counter() - started,
    )
    return {
        "best_epoch": training_result.best_epoch,
        "final_epoch": train_config.epochs,
        "representative_epochs": list(latent_result.representative_epochs),
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
    if any(epoch > train.epochs for epoch in analysis.representative_epochs):
        get_logger("run").warning(
            "Representative epochs beyond the requested training duration will be "
            "ignored; the final epoch is always included."
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
