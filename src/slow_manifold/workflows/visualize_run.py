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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from slow_manifold.analysis import AnalysisConfig, analyze_checkpoints
from slow_manifold.config import dump_yaml, load_yaml
from slow_manifold.models import Rank2CTRNNConfig
from slow_manifold.tasks import (
    IntervalCategorizationConfig,
    IntervalCategorizationTask,
)
from slow_manifold.utils import get_logger, setup_run_logging
from slow_manifold.visualization import (
    plot_representative_outputs,
    plot_training_curves,
    render_latent_vector_field_collection,
)

_EPOCH_CHECKPOINT = re.compile(r"epoch-(\d{6})\.pt")


def rerun_rank2_visualization(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    coordinate_bounds: Sequence[float] | None = None,
    grid_points: int | None = None,
    representative_epochs: Sequence[int] | None = None,
    arrow_stride: int | None = None,
) -> Path:
    """Recompute diagnostics if needed and re-render figures for one run.

    Analysis-affecting overrides (``coordinate_bounds``, ``grid_points``)
    force the latent dynamics to be recomputed from the stored checkpoints;
    otherwise the stored ``diagnostics/latent_dynamics.npz`` is reused.
    Figures are written to ``output_dir`` (default: the run's ``figures/``).
    Every event is appended to the run's ``run.log``.
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
    model_config = Rank2CTRNNConfig.from_mapping(resolved["components"]["model"])
    task_config = IntervalCategorizationConfig.from_mapping(
        resolved["components"]["task"], dt=model_config.dt
    )
    analysis_mapping = deepcopy(resolved["components"]["analysis"])
    visualization_mapping = deepcopy(resolved["components"]["visualization"])
    applied_overrides: dict[str, Any] = {}
    if coordinate_bounds is not None:
        analysis_mapping["coordinate_bounds"] = list(coordinate_bounds)
        applied_overrides["coordinate_bounds"] = list(coordinate_bounds)
    if grid_points is not None:
        analysis_mapping["grid_points"] = grid_points
        applied_overrides["grid_points"] = grid_points
    if arrow_stride is not None:
        visualization_mapping["arrow_stride"] = arrow_stride
        applied_overrides["arrow_stride"] = arrow_stride
    analysis_config = AnalysisConfig.from_mapping(analysis_mapping)

    recompute = (
        not data_path.is_file()
        or coordinate_bounds is not None
        or grid_points is not None
    )
    if recompute:
        checkpoint_paths = _epoch_checkpoint_paths(run_path)
        if not checkpoint_paths:
            raise RuntimeError(
                "Recompute requested but no epoch-*.pt checkpoints are stored "
                f"in {run_path / 'checkpoints'}"
            )
        validation_task = IntervalCategorizationTask(task_config, split="validation")
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
        representatives = list(result.representative_epochs)
    else:
        representatives = _stored_representative_epochs(
            diagnostics_dir / "latent_dynamics.yaml", data_path
        )
    representatives = _select_representatives(
        representatives, representative_epochs, data_path
    )

    figures_dir = (
        run_path / "figures"
        if output_dir is None
        else Path(output_dir).resolve()
    )
    started_figures = time.perf_counter()
    plot_training_curves(
        metrics_path,
        figures_dir / "loss_and_gradient.png",
        representative_epochs=representatives,
        figure_size=visualization_mapping["training_curves_figure_size"],
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
        dt=model_config.dt,
        panel_width=float(visualization_mapping["outputs_panel_width"]),
        panel_height=float(visualization_mapping["outputs_panel_height"]),
        decision_band_logit_half_width=float(
            visualization_mapping.get("decision_band_logit_half_width", 1.0)
        ),
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
        speed_floor=analysis_config.speed_floor,
        figure_size=visualization_mapping["vector_field_figure_size"],
        dpi=int(visualization_mapping["vector_field_dpi"]),
        fps=int(visualization_mapping["movie_fps"]),
        codec=str(visualization_mapping["movie_codec"]),
        arrow_stride=int(visualization_mapping["arrow_stride"]),
        decision_band_logit_half_width=float(
            visualization_mapping.get("decision_band_logit_half_width", 1.0)
        ),
    )
    logger.info(
        "vector field artifacts written snapshots=%d movie=latent_vector_field.mp4 "
        "elapsed_seconds=%.2fs",
        len(vector_fields["snapshots"]),
        time.perf_counter() - started_figures,
    )

    if applied_overrides:
        _record_overrides(diagnostics_dir, applied_overrides)
    logger.info(
        "visualization rerun complete elapsed_seconds=%.1fs figures=%s",
        time.perf_counter() - started,
        figures_dir,
    )
    return figures_dir


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


def _stored_representative_epochs(
    metadata_path: Path, data_path: Path
) -> list[int]:
    """Representative epochs recorded next to the stored diagnostics."""
    if metadata_path.is_file():
        metadata = load_yaml(metadata_path)
        epochs = metadata.get("representative_epochs")
        if epochs:
            return [int(epoch) for epoch in epochs]
    with np.load(data_path) as data:
        return [int(data["epochs"][-1])]


def _select_representatives(
    stored: Sequence[int],
    requested: Sequence[int] | None,
    data_path: Path,
) -> list[int]:
    """Keep requested epochs that exist in the data; always end with the last."""
    with np.load(data_path) as data:
        available = {int(epoch) for epoch in data["epochs"]}
        final_epoch = int(data["epochs"][-1])
    if requested is None:
        selected = [int(epoch) for epoch in stored if int(epoch) in available]
    else:
        selected = [int(epoch) for epoch in requested if int(epoch) in available]
    if final_epoch not in selected:
        selected.append(final_epoch)
    return selected


def _format_bounds(result: Any) -> str:
    """Short bounds display for the log line."""
    with np.load(result.data_path) as data:
        bounds = data["bounds"]
    return "[" + ", ".join(f"{value:.3g}" for value in bounds) + "]"


def _record_overrides(
    diagnostics_dir: Path, applied: Mapping[str, Any]
) -> None:
    """Leave a trace of non-default analysis/visualization overrides."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(
        {
            "applied_at_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "visualize_rerun",
            "overrides": dict(applied),
        },
        diagnostics_dir / "visualize_rerun.yaml",
    )
