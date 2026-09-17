"""Dinc-style training and latent-dynamics figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import imageio_ffmpeg
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LogNorm, Normalize, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Output and decision-band colors: class -1 / short (T < T_c) is red and
# class +1 / long (T > T_c) is green, mirroring the Dinc-style panels. The
# trajectory overlay uses a separate palette below to remain visible on maps.
_CLASS_SHORT_COLOR = "tab:red"
_CLASS_LONG_COLOR = "tab:green"
# Trajectories use a separate colorblind-safe palette and a white halo so
# they remain legible over both viridis speed maps and coolwarm Jacobian maps.
_TRAJECTORY_SHORT_COLOR = "#0072B2"
_TRAJECTORY_LONG_COLOR = "#E69F00"
_REPRODUCTION_TRAJECTORY_CMAP = "plasma"
_CUE_TRAJECTORY_COLOR = "#F0E442"
_REPRODUCTION_CUE_TRAJECTORY_COLOR = "tab:red"
_DEFAULT_TRAJECTORY_LINE_WIDTH = 1.5
# Default decision-band half-width in readout-logit units (Dinc convention:
# the band spans logits (-1, 1), i.e. sigma(-1)..sigma(1) for a sigmoid
# readout).  States inside the band are "not yet committed"; states outside
# are clearly committed to -1 or +1 and are left untinted (speed field only).
_DECISION_BAND_LOGIT_HALF_WIDTH = 1.0
_BAND_TINT_ALPHA = 0.35
_FIXED_POINT_COLOR = "tab:cyan"
_SLOW_POINT_COLOR = "tab:orange"
_GHOST_CANDIDATE_COLOR = "magenta"
_UNRESOLVED_MINIMUM_COLOR = "tab:gray"
_TRAJECTORY_PHASE_LINESTYLES = {
    "interval_encoding": "-",
    "delay": "--",
    "post_go_timing": "-.",
    "response": ":",
}
_HIGH_DIMENSIONAL_PHASE_COLORS = {
    "interval_encoding": "#0072B2",
    "delay": "#D55E00",
    "post_go_timing": "#009E73",
    "response": "#CC79A7",
}


def resolve_dpi_settings(config: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve PNG/movie DPI while preserving legacy config compatibility."""
    legacy = int(config.get("vector_field_dpi", 150))
    snapshot_dpi = int(config.get("snapshot_dpi", legacy))
    movie_dpi = int(config.get("movie_dpi", snapshot_dpi))
    if snapshot_dpi <= 0 or movie_dpi <= 0:
        raise ValueError("snapshot_dpi and movie_dpi must be positive")
    return snapshot_dpi, movie_dpi


def resolve_render_movies(config: Mapping[str, Any]) -> bool:
    """Resolve the configured movie switch with historical compatibility."""
    value = config.get("render_movies", True)
    if not isinstance(value, bool):
        raise ValueError("render_movies must be a boolean")
    return value


def _band_output_edge(logit_half_width: float) -> float:
    """Readout-output threshold |y| = tanh(w) for logit band half-width w."""
    return float(np.tanh(max(float(logit_half_width), 0.0)))


def _trial_class(target_row: np.ndarray) -> int:
    """Target class of one trial: +1 (long) if any positive target else -1."""
    return 1 if target_row.sum() > 0 else -1


def _class_color(class_value: int) -> str:
    return _CLASS_LONG_COLOR if class_value == 1 else _CLASS_SHORT_COLOR


def _trajectory_class_color(class_value: int) -> str:
    return (
        _TRAJECTORY_LONG_COLOR
        if class_value == 1
        else _TRAJECTORY_SHORT_COLOR
    )


def _trajectory_path_effects(line_width: float, *, alpha: float = 0.9):
    if line_width <= 0:
        raise ValueError("trajectory_line_width must be positive")
    return (
        patheffects.Stroke(
            linewidth=2.1 * line_width,
            foreground="white",
            alpha=alpha,
        ),
        patheffects.Normal(),
    )


def _cue_trajectory_color(data: Mapping[str, np.ndarray]) -> str:
    return (
        _REPRODUCTION_CUE_TRAJECTORY_COLOR
        if _is_reproduction(data)
        else _CUE_TRAJECTORY_COLOR
    )


def _cue_line_width(
    data: Mapping[str, np.ndarray], trajectory_line_width: float
) -> float:
    scale = 1.15 if _is_reproduction(data) else 1.45
    return scale * trajectory_line_width


def _cue_path_effects(
    data: Mapping[str, np.ndarray], trajectory_line_width: float
):
    if _is_reproduction(data):
        return ()
    cue_width = _cue_line_width(data, trajectory_line_width)
    return (
        patheffects.Stroke(
            linewidth=1.75 * cue_width,
            foreground="black",
            alpha=0.9,
        ),
        patheffects.Normal(),
    )


def plot_training_curves(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    representative_epochs: Sequence[int],
    representative_labels: Mapping[int, str] | None = None,
    figure_size: Sequence[float],
    loss_y_scale: str = "log",
    dpi: int = 150,
) -> Path:
    """Plot loss and gradient norms against epoch on aligned panels."""
    rows = _read_metrics(metrics_path)
    epochs = np.array([int(row["epoch"]) for row in rows])
    train_loss = np.array([float(row["train_loss"]) for row in rows])
    validation_samples = [
        (int(row["epoch"]), float(row["validation_loss"]))
        for row in rows
        if row.get("validation_loss", "").strip()
    ]
    if not validation_samples:
        raise ValueError("metrics contain no validation_loss samples")
    validation_epochs = np.asarray(
        [epoch for epoch, _ in validation_samples], dtype=np.int64
    )
    validation_loss = np.asarray(
        [loss for _, loss in validation_samples], dtype=np.float64
    )
    recurrent_gradient = np.array(
        [float(row["recurrent_gradient_norm"]) for row in rows]
    )
    total_gradient = np.array([float(row["total_gradient_norm"]) for row in rows])
    _validate_y_scale(loss_y_scale, (train_loss, validation_loss), name="loss")

    figure, axes = plt.subplots(
        2, 1, figsize=tuple(figure_size), sharex=True, constrained_layout=True
    )
    axes[0].plot(epochs, train_loss, label="train loss", color="black")
    axes[0].plot(
        validation_epochs,
        validation_loss,
        label="validation loss",
        color="tab:blue",
    )
    axes[0].set_yscale(loss_y_scale)
    axes[0].set_ylabel(
        "masked MSE" if loss_y_scale == "linear" else "masked MSE (log scale)"
    )
    axes[0].legend()
    axes[1].semilogy(
        epochs,
        recurrent_gradient,
        label=r"$||\nabla_M L||_F + ||\nabla_N L||_F$",
        color="tab:red",
    )
    axes[1].semilogy(
        epochs,
        total_gradient,
        label="total gradient norm",
        color="tab:orange",
        alpha=0.7,
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("gradient norm")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
        for epoch in representative_epochs:
            axis.axvline(epoch, color="0.65", linewidth=0.8, linestyle=":")
    for epoch in representative_epochs:
        label = (representative_labels or {}).get(int(epoch))
        if label:
            axes[0].text(
                epoch,
                0.98,
                label,
                transform=axes[0].get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color="0.3",
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _validate_y_scale(
    scale: str, series: Sequence[np.ndarray], *, name: str
) -> None:
    if scale not in {"linear", "log"}:
        raise ValueError(f"{name}_y_scale must be 'linear' or 'log'")
    values = np.concatenate(tuple(np.ravel(values) for values in series))
    if not np.isfinite(values).all():
        raise ValueError(f"{name} values must be finite")
    if scale == "log" and np.any(values <= 0):
        raise ValueError(f"{name} values must be positive for a log y-axis")


def plot_representative_outputs(
    diagnostics_path: str | Path,
    output_path: str | Path,
    *,
    representative_epochs: Sequence[int],
    representative_labels: Mapping[int, str] | None = None,
    dt: float,
    panel_width: float,
    panel_height: float,
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
    response_threshold: float = 0.5,
    dpi: int = 150,
) -> Path:
    """Compare network and target outputs at selected epochs.

    When the diagnostics carry the input waveforms, each trial gets an extra
    schematic strip on top (Ramesan Fig. 1a style): the S1/S2/Go square-wave
    pulses as a black step line on one shared cue channel. The output panels
    below share that trial's time axis. Runs whose stored diagnostics predate the
    ``inputs`` field fall back to output-only panels.  A translucent band
    marks the readout-logit decision band (Dinc convention).
    """
    with np.load(diagnostics_path) as data:
        epochs = data["epochs"]
        prediction = data["prediction"]
        target = data["target"]
        valid_mask = data["valid_mask"]
        inputs = data["inputs"] if "inputs" in data.files else None
        task_name = str(data["task_name"].item()) if "task_name" in data.files else ""
        intervals = data["trial_interval"] if "trial_interval" in data.files else None
        response_steps = (
            data["trial_response_step"]
            if "trial_response_step" in data.files
            else None
        )

    indices = [_epoch_index(epochs, epoch) for epoch in representative_epochs]
    trials = target.shape[0]
    n_columns = len(indices)
    if task_name == "delayed_interval_reproduction":
        return _plot_reproduction_outputs(
            output_path=output_path,
            inputs=inputs,
            prediction=prediction,
            target=target,
            valid_mask=valid_mask,
            intervals=intervals,
            response_steps=response_steps,
            representative_epochs=representative_epochs,
            representative_labels=representative_labels,
            indices=indices,
            dt=dt,
            panel_width=panel_width,
            panel_height=panel_height,
            response_threshold=response_threshold,
            dpi=dpi,
        )
    if inputs is not None:
        return _plot_representative_outputs_with_inputs(
            diagnostics_path=diagnostics_path,
            output_path=output_path,
            inputs=inputs,
            prediction=prediction,
            target=target,
            valid_mask=valid_mask,
            representative_epochs=representative_epochs,
            representative_labels=representative_labels,
            indices=indices,
            trials=trials,
            n_columns=n_columns,
            dt=dt,
            panel_width=panel_width,
            panel_height=panel_height,
            decision_band_logit_half_width=decision_band_logit_half_width,
            dpi=dpi,
        )

    figure, axes = plt.subplots(
        trials,
        n_columns,
        figsize=(panel_width * n_columns, panel_height * trials),
        sharex="row",
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    band_edge = _band_output_edge(decision_band_logit_half_width)
    for column, (epoch, frame) in enumerate(zip(representative_epochs, indices)):
        for trial in range(trials):
            axis = axes[trial, column]
            _draw_output_trace(
                axis,
                target=target,
                prediction=prediction,
                valid_mask=valid_mask,
                frame=frame,
                trial=trial,
                dt=dt,
                band_edge=band_edge,
            )
            if trial == 0:
                axis.set_title(
                    _representative_title(epoch, representative_labels)
                )
            if column == 0:
                axis.set_ylabel(
                    f"class {_trial_class(target[trial, :, 0]):+d}\noutput"
                )
            if trial == trials - 1:
                axis.set_xlabel("time")
    axes[0, 0].legend(
        handles=_output_legend_handles(target, valid_mask, band_edge),
        labels=_output_legend_labels(target, valid_mask, band_edge),
        loc="best",
        fontsize=7,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _representative_title(
    epoch: int, labels: Mapping[int, str] | None
) -> str:
    label = (labels or {}).get(int(epoch))
    return f"{label}\nepoch {epoch}" if label else f"epoch {epoch}"


def _plot_reproduction_outputs(
    *,
    output_path: str | Path,
    inputs: np.ndarray | None,
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    intervals: np.ndarray | None,
    response_steps: np.ndarray | None,
    representative_epochs: Sequence[int],
    representative_labels: Mapping[int, str] | None,
    indices: Sequence[int],
    dt: float,
    panel_width: float,
    panel_height: float,
    response_threshold: float,
    dpi: int,
) -> Path:
    """Task-B output panels without categorization decision-band semantics."""
    trials = target.shape[0]
    columns = len(indices)
    figure, axes = plt.subplots(
        trials,
        columns,
        figsize=(panel_width * columns, panel_height * trials),
        sharex="row",
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for column, (epoch, frame) in enumerate(zip(representative_epochs, indices)):
        for trial in range(trials):
            axis = axes[trial, column]
            length = int(valid_mask[trial].sum())
            time = np.arange(length) * dt
            axis.step(
                time,
                target[trial, :length, 0],
                where="post",
                color="black",
                label="target",
            )
            axis.step(
                time,
                prediction[frame, trial, :length, 0],
                where="post",
                color="tab:blue",
                linestyle="--",
                label="output",
            )
            if inputs is not None:
                amplitude = max(float(np.max(np.abs(inputs[trial, :length]))), 1e-9)
                axis.step(
                    time,
                    -1.0 + 0.25 * inputs[trial, :length, 0] / amplitude,
                    where="post",
                    color="0.45",
                    linewidth=0.8,
                    label="cue (offset)",
                )
            if response_steps is not None:
                axis.axvline(
                    response_steps[trial] * dt,
                    color="0.5",
                    linestyle=":",
                    linewidth=0.8,
                )
            axis.axhline(
                response_threshold,
                color="tab:orange",
                linestyle=":",
                linewidth=0.8,
            )
            axis.set_ylim(-1.1, 1.1)
            axis.grid(alpha=0.2)
            if trial == 0:
                axis.set_title(
                    _representative_title(epoch, representative_labels)
                )
            if column == 0:
                label = (
                    f"T={intervals[trial]:g}\noutput"
                    if intervals is not None
                    else "output"
                )
                axis.set_ylabel(label)
            if trial == trials - 1:
                axis.set_xlabel("time")
    axes[0, 0].legend(fontsize=7, loc="best")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_representative_outputs_with_inputs(
    *,
    diagnostics_path: str | Path,
    output_path: str | Path,
    inputs: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    representative_epochs: Sequence[int],
    representative_labels: Mapping[int, str] | None,
    indices: Sequence[int],
    trials: int,
    n_columns: int,
    dt: float,
    panel_width: float,
    panel_height: float,
    decision_band_logit_half_width: float,
    dpi: int,
) -> Path:
    """Output panels per trial, preceded by a shared input-schematic strip."""
    from matplotlib.gridspec import GridSpec

    n_rows = trials * 2
    figure = plt.figure(
        figsize=(panel_width * n_columns, panel_height * trials * 1.75),
        constrained_layout=True,
    )
    grid = GridSpec(
        n_rows,
        n_columns,
        figure=figure,
        height_ratios=[0.55, 1.0] * trials,
        hspace=0.35,
    )
    band_edge = _band_output_edge(decision_band_logit_half_width)
    output_master: Any = None
    for trial in range(trials):
        length = int(valid_mask[trial].sum())
        time = np.arange(length) * dt
        input_axis = figure.add_subplot(grid[2 * trial, :])
        input_axis.tick_params(labelbottom=False)
        _draw_input_schematic(
            input_axis,
            inputs=inputs[trial, :length],
            time=time,
        )
        for column, (epoch, frame) in enumerate(
            zip(representative_epochs, indices)
        ):
            axis = figure.add_subplot(grid[2 * trial + 1, column])
            axis.sharex(input_axis)
            if output_master is None:
                output_master = axis
            else:
                axis.sharey(output_master)
            axis.tick_params(labelbottom=(trial == trials - 1))
            _draw_output_trace(
                axis,
                target=target,
                prediction=prediction,
                valid_mask=valid_mask,
                frame=frame,
                trial=trial,
                dt=dt,
                band_edge=band_edge,
            )
            if trial == 0:
                axis.set_title(
                    _representative_title(epoch, representative_labels)
                )
            if column == 0:
                axis.set_ylabel(
                    f"class {_trial_class(target[trial, :length, 0]):+d}\noutput"
                )
            if trial == trials - 1:
                axis.set_xlabel("time")
    assert output_master is not None
    output_master.legend(
        handles=_output_legend_handles(target, valid_mask, band_edge),
        labels=_output_legend_labels(target, valid_mask, band_edge),
        loc="best",
        fontsize=7,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _draw_input_schematic(
    axis,
    *,
    inputs: np.ndarray,
    time: np.ndarray,
) -> None:
    """Draw the shared scalar S1/S2/Go square-wave input."""
    if inputs.ndim != 2 or inputs.shape[1] != 1:
        raise ValueError("timing-task inputs must have one shared cue channel")
    channel_names = ("cue",)
    amplitude = float(np.max(np.abs(inputs))) if inputs.size else 1.0
    amplitude = max(amplitude, 1e-9)
    band = 1.0
    for channel, _name in enumerate(channel_names):
        offset = channel * band
        values = offset + inputs[:, channel] / amplitude * 0.8
        axis.step(time, values, where="post", color="black", linewidth=1.0)
    axis.set_ylim(-0.15, (len(channel_names) - 1) * band + 0.95)
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_yticks([channel * band + 0.4 for channel in range(len(channel_names))])
    axis.set_yticklabels(channel_names, fontsize=7)
    axis.grid(alpha=0.2)


def _draw_output_trace(
    axis,
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    frame: int,
    trial: int,
    dt: float,
    band_edge: float,
) -> None:
    """One output panel: decision-band halves, target and network output.

    The lower band half (output in ``(-band_edge, 0)``, leaning toward ``-1``)
    is tinted red and the upper half (output in ``(0, band_edge)``, leaning
    toward ``+1``) green — the ambiguous region inside which the readout is
    not yet committed; beyond the band the output is clearly ``-1``/``+1``.
    """
    length = int(valid_mask[trial].sum())
    time = np.arange(length) * dt
    class_value = _trial_class(target[trial, :length, 0])
    color = _class_color(class_value)
    axis.axhspan(-band_edge, 0.0, color=_CLASS_SHORT_COLOR, alpha=0.25, zorder=0)
    axis.axhspan(0.0, band_edge, color=_CLASS_LONG_COLOR, alpha=0.25, zorder=0)
    axis.step(
        time,
        target[trial, :length, 0],
        where="post",
        color=color,
        linestyle="-",
        label="_nolegend_",
    )
    axis.step(
        time,
        prediction[frame, trial, :length, 0],
        where="post",
        color=color,
        linestyle="--",
        label="_nolegend_",
    )
    axis.axhline(0, color="0.7", linewidth=0.7)
    axis.set_ylim(-1.1, 1.1)
    axis.grid(alpha=0.2)


def _output_legend_handles(
    target: np.ndarray,
    valid_mask: np.ndarray,
    band_edge: float,
) -> list[Line2D | Patch]:
    handles: list[Line2D | Patch] = [
        handle
        for class_value in _seen_classes(target, valid_mask)
        for handle in (
            Line2D(
                [],
                [],
                color=_class_color(class_value),
                linewidth=1.5,
                linestyle="-",
            ),
            Line2D(
                [],
                [],
                color=_class_color(class_value),
                linewidth=1.5,
                linestyle="--",
            ),
        )
    ]
    handles.append(
        Patch(facecolor=_CLASS_SHORT_COLOR, alpha=0.25, label="_nolegend_")
    )
    handles.append(
        Patch(facecolor=_CLASS_LONG_COLOR, alpha=0.25, label="_nolegend_")
    )
    return handles


def _output_legend_labels(
    target: np.ndarray,
    valid_mask: np.ndarray,
    band_edge: float,
) -> list[str]:
    labels: list[str] = []
    for class_value in _seen_classes(target, valid_mask):
        suffix = "long; T>T_c" if class_value == 1 else "short; T<T_c"
        labels.append(f"target (class {class_value:+d}; {suffix})")
        labels.append(f"output (class {class_value:+d}; {suffix})")
    labels.append(
        f"decision band $-1$ side (uncommitted; $y\\in(-{band_edge:.3g},0)$)"
    )
    labels.append(
        f"decision band $+1$ side (uncommitted; $y\\in(0,{band_edge:.3g})$) "
        f"[$|\\mathrm{{logit}}|<1$]"
    )
    return labels


def _seen_classes(target: np.ndarray, valid_mask: np.ndarray) -> list[int]:
    return sorted(
        {
            _trial_class(target[trial, : int(valid_mask[trial].sum()), 0])
            for trial in range(target.shape[0])
        }
    )


def render_latent_vector_field_collection(
    diagnostics_path: str | Path,
    output_dir: str | Path,
    *,
    representative_epochs: Sequence[int],
    representative_labels: Mapping[int, str] | None = None,
    speed_floor: float,
    figure_size: Sequence[float],
    dpi: int,
    movie_dpi: int | None = None,
    render_movie: bool = True,
    fps: int,
    codec: str,
    arrow_stride: int,
    arrow_color: str = "black",
    arrow_length_fraction: float = 0.0075,
    arrow_width: float = 0.0012,
    trajectory_line_width: float = _DEFAULT_TRAJECTORY_LINE_WIDTH,
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
    response_threshold: float = 0.5,
) -> dict[str, Path | list[Path] | None]:
    """Isolated folder with one snapshot per representative epoch + the MP4."""
    with np.load(diagnostics_path) as data:
        if int(data.get("coordinate_dimension", np.asarray(2))) >= 3:
            return {"snapshots": [], "movie": None}
        available = {int(epoch) for epoch in data["epochs"]}
    snapshot_epochs = [int(e) for e in representative_epochs if int(e) in available]

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshots: list[Path] = []
    for epoch in snapshot_epochs:
        path = plot_latent_vector_field_snapshot(
            diagnostics_path,
            destination / f"epoch-{epoch:06d}.png",
            epoch=epoch,
            representative_label=(representative_labels or {}).get(epoch),
            speed_floor=speed_floor,
            figure_size=figure_size,
            dpi=dpi,
            arrow_stride=arrow_stride,
            arrow_color=arrow_color,
            arrow_length_fraction=arrow_length_fraction,
            arrow_width=arrow_width,
            trajectory_line_width=trajectory_line_width,
            decision_band_logit_half_width=decision_band_logit_half_width,
            response_threshold=response_threshold,
        )
        snapshots.append(path)
    movie = None
    if render_movie:
        movie = create_latent_vector_field_movie(
            diagnostics_path,
            destination / "latent_vector_field.mp4",
            speed_floor=speed_floor,
            figure_size=figure_size,
            dpi=dpi if movie_dpi is None else movie_dpi,
            fps=fps,
            codec=codec,
            arrow_stride=arrow_stride,
            arrow_color=arrow_color,
            arrow_length_fraction=arrow_length_fraction,
            arrow_width=arrow_width,
            trajectory_line_width=trajectory_line_width,
            decision_band_logit_half_width=decision_band_logit_half_width,
            response_threshold=response_threshold,
        )
    return {"snapshots": snapshots, "movie": movie}


def render_latent_dynamics_collection(
    diagnostics_path: str | Path,
    output_dir: str | Path,
    *,
    representative_epochs: Sequence[int],
    representative_labels: Mapping[int, str] | None = None,
    speed_floor: float,
    figure_size: Sequence[float],
    dpi: int,
    movie_dpi: int | None = None,
    render_movie: bool = True,
    fps: int,
    codec: str,
    arrow_stride: int,
    arrow_color: str = "black",
    arrow_length_fraction: float = 0.0075,
    arrow_width: float = 0.0012,
    trajectory_line_width: float = _DEFAULT_TRAJECTORY_LINE_WIDTH,
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
    response_threshold: float = 0.5,
    spectral_abscissa_limit: float | None = None,
    max_trajectories: int = 64,
    trajectory_3d_alpha: float = 0.62,
    trajectory_3d_cue_alpha_scale: float = 0.55,
    trajectory_3d_cue_line_width_scale: float = 0.8,
    trajectory_3d_view_elevation: float = 26.0,
    trajectory_3d_view_azimuth: float = -68.0,
    single_trajectory_enabled: bool = True,
    single_trajectory_figure_size: Sequence[float] = (7.0, 6.0),
    single_trajectory_interval_quantile: float = 0.5,
) -> dict[str, Path | list[Path] | None]:
    """Render synchronized vector-field/Jacobian panels and their MP4."""
    with np.load(diagnostics_path) as stored:
        coordinate_dimension = int(
            stored.get("coordinate_dimension", np.asarray(2))
        )
    if coordinate_dimension >= 3:
        return render_high_dimensional_dynamics_collection(
            diagnostics_path,
            output_dir,
            representative_epochs=representative_epochs,
            representative_labels=representative_labels,
            figure_size=figure_size,
            dpi=dpi,
            movie_dpi=movie_dpi,
            render_movie=render_movie,
            fps=fps,
            codec=codec,
            trajectory_line_width=trajectory_line_width,
            max_trajectories=max_trajectories,
            spectral_abscissa_limit=spectral_abscissa_limit,
            trajectory_alpha=trajectory_3d_alpha,
            cue_alpha_scale=trajectory_3d_cue_alpha_scale,
            cue_line_width_scale=trajectory_3d_cue_line_width_scale,
            view_elevation=trajectory_3d_view_elevation,
            view_azimuth=trajectory_3d_view_azimuth,
            single_trajectory_enabled=single_trajectory_enabled,
            single_trajectory_figure_size=single_trajectory_figure_size,
            single_trajectory_interval_quantile=(
                single_trajectory_interval_quantile
            ),
        )
    data = _load_vector_field_data(diagnostics_path)
    available = {int(epoch) for epoch in data["epochs"]}
    snapshot_epochs = [
        int(epoch)
        for epoch in representative_epochs
        if int(epoch) in available
    ]

    log_speed = _log_speed(data["speed"], speed_floor)
    speed_norm = Normalize(vmin=float(log_speed.min()), vmax=float(log_speed.max()))
    jacobian_norm = _spectral_abscissa_norm(
        data["jacobian_spectral_abscissa"], spectral_abscissa_limit
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshots = [
        _plot_latent_dynamics_snapshot(
            data=data,
            output_path=destination / f"epoch-{epoch:06d}.png",
            epoch=epoch,
            representative_label=(representative_labels or {}).get(epoch),
            log_speed=log_speed,
            speed_norm=speed_norm,
            jacobian_norm=jacobian_norm,
            speed_floor=speed_floor,
            figure_size=figure_size,
            dpi=dpi,
            arrow_stride=arrow_stride,
            arrow_color=arrow_color,
            arrow_length_fraction=arrow_length_fraction,
            arrow_width=arrow_width,
            trajectory_line_width=trajectory_line_width,
            decision_band_logit_half_width=decision_band_logit_half_width,
            response_threshold=response_threshold,
        )
        for epoch in snapshot_epochs
    ]
    movie = None
    if render_movie:
        movie = _create_latent_dynamics_movie(
            data=data,
            output_path=destination / "latent_dynamics.mp4",
            log_speed=log_speed,
            speed_norm=speed_norm,
            jacobian_norm=jacobian_norm,
            speed_floor=speed_floor,
            figure_size=figure_size,
            dpi=dpi if movie_dpi is None else movie_dpi,
            fps=fps,
            codec=codec,
            arrow_stride=arrow_stride,
            arrow_color=arrow_color,
            arrow_length_fraction=arrow_length_fraction,
            arrow_width=arrow_width,
            trajectory_line_width=trajectory_line_width,
            decision_band_logit_half_width=decision_band_logit_half_width,
            response_threshold=response_threshold,
        )
    return {"snapshots": snapshots, "movie": movie}


def render_high_dimensional_dynamics_collection(
    diagnostics_path: str | Path,
    output_dir: str | Path,
    *,
    representative_epochs: Sequence[int],
    representative_labels: Mapping[int, str] | None = None,
    figure_size: Sequence[float] = (15.0, 5.0),
    dpi: int = 180,
    movie_dpi: int | None = None,
    render_movie: bool = True,
    fps: int = 5,
    codec: str = "libx264",
    trajectory_line_width: float = _DEFAULT_TRAJECTORY_LINE_WIDTH,
    max_trajectories: int = 64,
    spectral_abscissa_limit: float | None = None,
    trajectory_alpha: float = 0.62,
    cue_alpha_scale: float = 0.55,
    cue_line_width_scale: float = 0.8,
    view_elevation: float = 26.0,
    view_azimuth: float = -68.0,
    single_trajectory_enabled: bool = True,
    single_trajectory_figure_size: Sequence[float] = (7.0, 6.0),
    single_trajectory_interval_quantile: float = 0.5,
) -> dict[str, Path | list[Path] | None]:
    """Render 3-D trajectories and trajectory-seeded full-state slow points."""
    if max_trajectories <= 0:
        raise ValueError("max_trajectories must be positive")
    if not isinstance(single_trajectory_enabled, bool):
        raise ValueError("single_trajectory_enabled must be a boolean")
    if not 0 < trajectory_alpha <= 1:
        raise ValueError("trajectory_3d_alpha must lie in (0, 1]")
    if not 0 <= cue_alpha_scale <= 1 or cue_line_width_scale <= 0:
        raise ValueError(
            "trajectory cue alpha scale must lie in [0, 1] and width scale positive"
        )
    if not 0 <= single_trajectory_interval_quantile <= 1:
        raise ValueError("single_trajectory_interval_quantile must lie in [0, 1]")
    data = _load_vector_field_data(diagnostics_path)
    if int(data["coordinate_dimension"]) != 3:
        raise ValueError("high-dimensional display diagnostics must be 3-D")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    available = {int(epoch) for epoch in data["epochs"]}
    snapshot_epochs = [
        int(epoch) for epoch in representative_epochs if int(epoch) in available
    ]
    refined = bool(data.get("slow_point_search_enabled", np.asarray(False)))
    if refined:
        snapshot_indices = np.asarray(
            [_epoch_index(data["epochs"], epoch) for epoch in snapshot_epochs]
        )
        selected = data["slow_point_accepted"] & np.isin(
            data["slow_point_epoch_index"], snapshot_indices
        )
        q_values = data["slow_point_q"]
        q_norm = _robust_positive_log_norm(q_values[selected])
        spectral_norm = _spectral_abscissa_norm(
            data["slow_point_spectral_abscissa"][selected],
            spectral_abscissa_limit,
        )
    else:
        q_values = np.maximum(data["neighborhood_q"], np.finfo(float).tiny)
        q_norm = _robust_positive_log_norm(q_values)
        spectral_norm = _spectral_abscissa_norm(
            data["neighborhood_spectral_abscissa"], spectral_abscissa_limit
        )

    snapshots: list[Path] = []
    for epoch in snapshot_epochs:
        frame_index = _epoch_index(data["epochs"], epoch)
        figure = _high_dimensional_figure(figure_size)
        _draw_high_dimensional_frame(
            figure,
            data=data,
            frame_index=frame_index,
            epoch=epoch,
            representative_label=(representative_labels or {}).get(epoch),
            q_values=q_values,
            q_norm=q_norm,
            spectral_norm=spectral_norm,
            trajectory_line_width=trajectory_line_width,
            max_trajectories=max_trajectories,
            refined_slow_points=refined,
            trajectory_alpha=trajectory_alpha,
            cue_alpha_scale=cue_alpha_scale,
            cue_line_width_scale=cue_line_width_scale,
            view_elevation=view_elevation,
            view_azimuth=view_azimuth,
        )
        path = destination / f"epoch-{epoch:06d}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        snapshots.append(path)

    movie_path = None
    if render_movie:
        movie_path = destination / "latent_dynamics.mp4"
        figure = _high_dimensional_figure(figure_size)
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(
            fps=fps,
            codec=codec,
            extra_args=[
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )
        with writer.saving(
            figure,
            movie_path,
            dpi=dpi if movie_dpi is None else movie_dpi,
        ):
            for frame_index, epoch in enumerate(data["epochs"]):
                figure.clear()
                _draw_high_dimensional_frame(
                    figure,
                    data=data,
                    frame_index=frame_index,
                    epoch=int(epoch),
                    representative_label=None,
                    q_values=q_values,
                    q_norm=q_norm,
                    spectral_norm=spectral_norm,
                    trajectory_line_width=trajectory_line_width,
                    max_trajectories=max_trajectories,
                    refined_slow_points=refined,
                    trajectory_alpha=trajectory_alpha,
                    cue_alpha_scale=cue_alpha_scale,
                    cue_line_width_scale=cue_line_width_scale,
                    view_elevation=view_elevation,
                    view_azimuth=view_azimuth,
                )
                writer.grab_frame()
        plt.close(figure)
    single_trajectory_path = None
    if single_trajectory_enabled and snapshot_epochs:
        epoch = snapshot_epochs[-1]
        frame_index = _epoch_index(data["epochs"], epoch)
        single_trajectory_path = _plot_single_trajectory_schematic(
            data=data,
            output_path=(
                destination
                / "single_trajectory"
                / f"epoch-{epoch:06d}.png"
            ),
            frame_index=frame_index,
            epoch=epoch,
            representative_label=(representative_labels or {}).get(epoch),
            figure_size=single_trajectory_figure_size,
            dpi=dpi,
            line_width=1.35 * trajectory_line_width,
            interval_quantile=single_trajectory_interval_quantile,
            cue_alpha_scale=cue_alpha_scale,
            cue_line_width_scale=cue_line_width_scale,
            view_elevation=view_elevation,
            view_azimuth=view_azimuth,
        )
    return {
        "snapshots": snapshots,
        "movie": movie_path,
        "single_trajectory": single_trajectory_path,
    }


def _high_dimensional_figure(figure_size: Sequence[float]):
    return plt.figure(figsize=tuple(figure_size), constrained_layout=True)


def _draw_high_dimensional_frame(
    figure,
    *,
    data: Mapping[str, np.ndarray],
    frame_index: int,
    epoch: int,
    representative_label: str | None,
    q_values: np.ndarray,
    q_norm: Normalize,
    spectral_norm: Normalize,
    trajectory_line_width: float,
    max_trajectories: int,
    refined_slow_points: bool,
    trajectory_alpha: float,
    cue_alpha_scale: float,
    cue_line_width_scale: float,
    view_elevation: float,
    view_azimuth: float,
) -> None:
    axes = [figure.add_subplot(1, 3, index, projection="3d") for index in (1, 2, 3)]
    title_suffix = (
        f"{representative_label} — epoch {epoch}"
        if representative_label
        else f"epoch {epoch}"
    )
    selected_trials = _selected_trajectory_indices(data, max_trajectories)
    _draw_3d_trajectories(
        axes[0],
        data=data,
        frame_index=frame_index,
        selected_trials=selected_trials,
        line_width=trajectory_line_width,
        alpha=trajectory_alpha,
        phase_styles=True,
        cue_alpha_scale=cue_alpha_scale,
        cue_line_width_scale=cue_line_width_scale,
        halo_alpha=0.45,
    )
    axes[0].set_title(f"task trajectories\n{title_suffix}")

    for axis in axes[1:]:
        _draw_3d_trajectories(
            axis,
            data=data,
            frame_index=frame_index,
            selected_trials=selected_trials,
            line_width=0.55 * trajectory_line_width,
            alpha=0.10,
            phase_styles=False,
            cue_alpha_scale=cue_alpha_scale,
            cue_line_width_scale=cue_line_width_scale,
            halo_alpha=0.25,
        )
    if refined_slow_points:
        frame_points = data["slow_point_epoch_index"] == frame_index
        selected = frame_points & data["slow_point_accepted"]
        coordinates = data["slow_point_coordinates"]
        q = q_values
        spectral = data["slow_point_spectral_abscissa"]
        fixed = selected & data["slow_point_is_fixed"]
        colored_q = selected & ~fixed & (q > 0)
    else:
        coordinates = data["neighborhood_coordinates"][frame_index]
        selected = data["neighborhood_low_q_mask"][frame_index]
        q = q_values[frame_index]
        spectral = data["neighborhood_spectral_abscissa"][frame_index]
        fixed = np.zeros_like(selected)
        colored_q = selected
    if np.any(colored_q):
        slow_scatter = axes[1].scatter(
            *coordinates[colored_q].T,
            c=q[colored_q],
            cmap="viridis_r",
            norm=q_norm,
            s=12,
            alpha=0.85,
            depthshade=False,
        )
        figure.colorbar(
            slow_scatter,
            ax=axes[1],
            shrink=0.58,
            pad=0.02,
            label=r"$q_x=\frac{1}{2}\|F_x\|^2$",
        )
    if np.any(fixed):
        axes[1].scatter(
            *coordinates[fixed].T,
            facecolors="none",
            edgecolors="cyan",
            marker="D",
            s=24,
            linewidths=0.9,
            depthshade=False,
        )
    if not np.any(selected):
        axes[1].text2D(0.3, 0.5, "no selected samples", transform=axes[1].transAxes)
    if refined_slow_points:
        stationary = selected & data["slow_point_optimization_converged"]
        axes[1].set_title(
            "optimized low-q endpoints\n"
            f"q-accepted {int(selected.sum())}; stationary {int(stationary.sum())}"
        )
        spectral_selected = selected
    else:
        threshold = float(data["neighborhood_low_q_threshold"][frame_index])
        axes[1].set_title(f"legacy sampled low-q region\nq ≤ {threshold:.2e}")
        spectral_selected = data["neighborhood_near_zero_mask"][frame_index]

    if np.any(spectral_selected):
        spectrum_scatter = axes[2].scatter(
            *coordinates[spectral_selected].T,
            c=spectral[spectral_selected],
            cmap="coolwarm",
            norm=spectral_norm,
            s=12,
            alpha=0.85,
            depthshade=False,
        )
        figure.colorbar(
            spectrum_scatter,
            ax=axes[2],
            shrink=0.58,
            pad=0.02,
            label=r"$\max_i\,\mathrm{Re}\,\lambda_i(J_x)$",
        )
    else:
        axes[2].text2D(
            0.27, 0.5, "no near-zero samples", transform=axes[2].transAxes
        )
    axes[2].set_title(
        "same slow points: spectral abscissa"
        if refined_slow_points
        else "legacy near-zero spectral abscissa"
    )

    labels = [str(value) for value in data["coordinate_labels"]]
    bounds = (
        data["display_bounds_by_epoch"][frame_index]
        if "display_bounds_by_epoch" in data
        else data["display_bounds"]
    )
    for axis in axes:
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.set_zlabel(labels[2])
        axis.set_xlim(*bounds[0])
        axis.set_ylim(*bounds[1])
        axis.set_zlim(*bounds[2])
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=view_elevation, azim=view_azimuth)
    axes[0].legend(
        handles=_high_dimensional_legend_handles(
            data,
            trajectory_line_width,
            cue_line_width_scale=cue_line_width_scale,
        ),
        loc="upper left",
        fontsize=6,
        frameon=False,
    )


def _plot_single_trajectory_schematic(
    *,
    data: Mapping[str, np.ndarray],
    output_path: Path,
    frame_index: int,
    epoch: int,
    representative_label: str | None,
    figure_size: Sequence[float],
    dpi: int,
    line_width: float,
    interval_quantile: float,
    cue_alpha_scale: float,
    cue_line_width_scale: float,
    view_elevation: float,
    view_azimuth: float,
) -> Path:
    trial = _representative_trajectory_index(data, interval_quantile)
    figure = plt.figure(figsize=tuple(figure_size), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    _draw_3d_trajectories(
        axis,
        data=data,
        frame_index=frame_index,
        selected_trials=np.asarray([trial]),
        line_width=line_width,
        alpha=0.96,
        phase_styles=True,
        cue_alpha_scale=min(1.0, 1.25 * cue_alpha_scale),
        cue_line_width_scale=cue_line_width_scale,
        halo_alpha=0.8,
    )
    length = int(data["trajectory_valid_mask"][trial].sum())
    points = data["trajectory"][frame_index, trial, :length]
    bounds = _single_trajectory_bounds(points, padding_fraction=0.08)
    labels = [str(value) for value in data["coordinate_labels"]]
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_zlabel(labels[2])
    axis.set_xlim(*bounds[0])
    axis.set_ylim(*bounds[1])
    axis.set_zlim(*bounds[2])
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=view_elevation, azim=view_azimuth)
    interval = float(data["trajectory_trial_interval"][trial])
    delay = float(data["trajectory_trial_delay"][trial])
    label = f"{representative_label} — " if representative_label else ""
    axis.set_title(
        f"single task trajectory\n{label}epoch {epoch}; T={interval:g}, delay={delay:g}"
    )
    axis.legend(
        handles=_high_dimensional_legend_handles(
            data,
            line_width,
            cue_line_width_scale=cue_line_width_scale,
        ),
        loc="upper left",
        fontsize=7,
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _representative_trajectory_index(
    data: Mapping[str, np.ndarray], interval_quantile: float
) -> int:
    intervals = np.asarray(data["trajectory_trial_interval"], dtype=float)
    target = float(np.quantile(intervals, interval_quantile))
    distances = np.abs(intervals - target)
    return int(np.flatnonzero(distances == distances.min())[0])


def _single_trajectory_bounds(
    points: np.ndarray, *, padding_fraction: float
) -> np.ndarray:
    minima = points.min(axis=0)
    maxima = points.max(axis=0)
    center = 0.5 * (minima + maxima)
    span = maxima - minima
    minimum_span = max(float(span.max()) * 0.04, 1.0e-6)
    half_span = 0.5 * np.maximum(span, minimum_span)
    half_span *= 1.0 + 2.0 * padding_fraction
    return np.column_stack((center - half_span, center + half_span))


def _selected_trajectory_indices(
    data: Mapping[str, np.ndarray], maximum: int
) -> np.ndarray:
    count = int(data["trajectory"].shape[1])
    if count <= maximum:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=int))


def _draw_3d_trajectories(
    axis,
    *,
    data: Mapping[str, np.ndarray],
    frame_index: int,
    selected_trials: np.ndarray,
    line_width: float,
    alpha: float,
    phase_styles: bool,
    cue_alpha_scale: float = 1.0,
    cue_line_width_scale: float = 1.0,
    halo_alpha: float = 0.9,
) -> None:
    trajectories = data["trajectory"][frame_index]
    target = data["trajectory_target"]
    valid_mask = data["trajectory_valid_mask"]
    for trial in selected_trials:
        length = int(valid_mask[trial].sum())
        points = trajectories[trial, :length]
        color = _high_dimensional_trajectory_color(data, target, valid_mask, int(trial))
        if not phase_styles:
            axis.plot(*points.T, color=color, linewidth=line_width, alpha=alpha)
            continue
        s1_step = int(data["trajectory_trial_s1_step"][trial])
        if s1_step > 0:
            baseline = points[: min(s1_step + 1, length)]
            axis.plot(
                *baseline.T,
                color="tab:gray",
                linewidth=0.55 * line_width,
                alpha=0.35 * alpha,
            )
        for phase, start, stop in _high_dimensional_phase_ranges(
            data, trial=int(trial), length=length
        ):
            segment = points[start : stop + 1]
            if segment.shape[0] >= 2:
                axis.plot(
                    *segment.T,
                    color=_HIGH_DIMENSIONAL_PHASE_COLORS[phase],
                    linewidth=line_width,
                    linestyle=_TRAJECTORY_PHASE_LINESTYLES[phase],
                    alpha=alpha,
                    path_effects=_trajectory_path_effects(
                        line_width, alpha=halo_alpha
                    ),
                )
        for start, stop in _cue_driven_point_ranges(
            data["trajectory_inputs"][trial, :length]
        ):
            segment = points[start:stop]
            if segment.shape[0] >= 2:
                axis.plot(
                    *segment.T,
                    color=_cue_trajectory_color(data),
                    linewidth=(
                        cue_line_width_scale
                        * _cue_line_width(data, line_width)
                    ),
                    alpha=alpha * cue_alpha_scale,
                    path_effects=_cue_path_effects(data, line_width),
                    zorder=4,
                )
        axis.scatter(
            *points[-1],
            color=color,
            marker=(
                "o"
                if _is_reproduction(data)
                else (
                    "^"
                    if _trial_class(target[trial, :length, 0]) > 0
                    else "v"
                )
            ),
            s=max(8.0, 8.0 * line_width**2),
            edgecolors="white",
            linewidths=0.4 * line_width,
            alpha=alpha,
            depthshade=False,
            zorder=5,
        )


def _high_dimensional_phase_ranges(
    data: Mapping[str, np.ndarray], *, trial: int, length: int
) -> list[tuple[str, int, int]]:
    boundaries = (
        int(data["trajectory_trial_s1_step"][trial]),
        int(data["trajectory_trial_s2_step"][trial]),
        int(data["trajectory_trial_go_step"][trial]),
        int(data["trajectory_trial_response_step"][trial]),
        length - 1,
    )
    return [
        (phase, max(0, start), min(stop, length - 1))
        for phase, start, stop in zip(
            _TRAJECTORY_PHASE_LINESTYLES, boundaries[:-1], boundaries[1:]
        )
        if start < length and stop > start
    ]


def _high_dimensional_trajectory_color(
    data: Mapping[str, np.ndarray],
    target: np.ndarray,
    valid_mask: np.ndarray,
    trial: int,
):
    if not _is_reproduction(data):
        length = int(valid_mask[trial].sum())
        return _trajectory_class_color(_trial_class(target[trial, :length, 0]))
    values = data["trajectory_trial_interval"]
    span = max(float(np.ptp(values)), 1.0)
    return plt.get_cmap(_REPRODUCTION_TRAJECTORY_CMAP)(
        (float(values[trial]) - float(values.min())) / span
    )


def _high_dimensional_legend_handles(
    data: Mapping[str, np.ndarray], line_width: float,
    *,
    cue_line_width_scale: float = 1.0,
) -> list[Line2D]:
    if _is_reproduction(data):
        condition_handles = [
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markerfacecolor="tab:purple",
                label="endpoint color: T",
            )
        ]
    else:
        condition_handles = [
            Line2D(
                [],
                [],
                marker="v",
                color="none",
                markerfacecolor=_TRAJECTORY_SHORT_COLOR,
                label="short endpoint",
            ),
            Line2D(
                [],
                [],
                marker="^",
                color="none",
                markerfacecolor=_TRAJECTORY_LONG_COLOR,
                label="long endpoint",
            ),
        ]
    phase_labels = {
        "interval_encoding": "interval encoding",
        "delay": "delay",
        "post_go_timing": "post-Go timing",
        "response": "response",
    }
    return condition_handles + [
        Line2D(
            [],
            [],
            color=_HIGH_DIMENSIONAL_PHASE_COLORS[phase],
            linestyle=style,
            linewidth=line_width,
            label=phase_labels[phase],
        )
        for phase, style in _TRAJECTORY_PHASE_LINESTYLES.items()
    ] + [
        Line2D(
            [],
            [],
            color=_cue_trajectory_color(data),
            linewidth=(
                cue_line_width_scale * _cue_line_width(data, line_width)
            ),
            label=r"cue on ($u\ne0$)",
        )
    ]


def _plot_latent_dynamics_snapshot(
    *,
    data: Mapping[str, np.ndarray],
    output_path: Path,
    epoch: int,
    representative_label: str | None,
    log_speed: np.ndarray,
    speed_norm: Normalize,
    jacobian_norm: Normalize,
    speed_floor: float,
    figure_size: Sequence[float],
    dpi: int,
    arrow_stride: int,
    arrow_color: str,
    arrow_length_fraction: float,
    arrow_width: float,
    trajectory_line_width: float,
    decision_band_logit_half_width: float,
    response_threshold: float,
) -> Path:
    frame_index = _epoch_index(data["epochs"], epoch)
    figure, axes = _latent_dynamics_figure(
        figure_size=figure_size,
        speed_norm=speed_norm,
        jacobian_norm=jacobian_norm,
    )
    _draw_latent_dynamics_frame(
        axes,
        data=data,
        frame_index=frame_index,
        epoch=epoch,
        representative_label=representative_label,
        log_speed=log_speed,
        speed_norm=speed_norm,
        jacobian_norm=jacobian_norm,
        speed_floor=speed_floor,
        arrow_stride=arrow_stride,
        arrow_color=arrow_color,
        arrow_length_fraction=arrow_length_fraction,
        arrow_width=arrow_width,
        trajectory_line_width=trajectory_line_width,
        decision_band_logit_half_width=decision_band_logit_half_width,
        response_threshold=response_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _create_latent_dynamics_movie(
    *,
    data: Mapping[str, np.ndarray],
    output_path: Path,
    log_speed: np.ndarray,
    speed_norm: Normalize,
    jacobian_norm: Normalize,
    speed_floor: float,
    figure_size: Sequence[float],
    dpi: int,
    fps: int,
    codec: str,
    arrow_stride: int,
    arrow_color: str,
    arrow_length_fraction: float,
    arrow_width: float,
    trajectory_line_width: float,
    decision_band_logit_half_width: float,
    response_threshold: float,
) -> Path:
    figure, axes = _latent_dynamics_figure(
        figure_size=figure_size,
        speed_norm=speed_norm,
        jacobian_norm=jacobian_norm,
    )
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = FFMpegWriter(
        fps=fps,
        codec=codec,
        extra_args=[
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, output_path, dpi=dpi):
        for frame_index, epoch in enumerate(data["epochs"]):
            for axis in axes:
                axis.clear()
            _draw_latent_dynamics_frame(
                axes,
                data=data,
                frame_index=frame_index,
                epoch=int(epoch),
                log_speed=log_speed,
                speed_norm=speed_norm,
                jacobian_norm=jacobian_norm,
                speed_floor=speed_floor,
                arrow_stride=arrow_stride,
                arrow_color=arrow_color,
                arrow_length_fraction=arrow_length_fraction,
                arrow_width=arrow_width,
                trajectory_line_width=trajectory_line_width,
                decision_band_logit_half_width=decision_band_logit_half_width,
                response_threshold=response_threshold,
            )
            writer.grab_frame()
    plt.close(figure)
    return output_path


def _latent_dynamics_figure(
    *,
    figure_size: Sequence[float],
    speed_norm: Normalize,
    jacobian_norm: Normalize,
):
    figure, axes = plt.subplots(
        1, 2, figsize=tuple(figure_size), constrained_layout=True
    )
    figure.colorbar(
        ScalarMappable(norm=speed_norm, cmap="viridis"),
        ax=axes[0],
        label=r"$\log_{10}\,||\tau F_\kappa||_2$",
    )
    figure.colorbar(
        ScalarMappable(norm=jacobian_norm, cmap="coolwarm"),
        ax=axes[1],
        label=r"$\max_i\,\mathrm{Re}\,\lambda_i(J_\kappa)$",
    )
    return figure, axes


def _draw_latent_dynamics_frame(
    axes,
    *,
    data: Mapping[str, np.ndarray],
    frame_index: int,
    epoch: int,
    representative_label: str | None = None,
    log_speed: np.ndarray,
    speed_norm: Normalize,
    jacobian_norm: Normalize,
    speed_floor: float,
    arrow_stride: int,
    arrow_color: str,
    arrow_length_fraction: float,
    arrow_width: float,
    trajectory_line_width: float,
    decision_band_logit_half_width: float,
    response_threshold: float,
) -> None:
    epoch_title = (
        f"{representative_label} — epoch {epoch}"
        if representative_label
        else f"epoch {epoch}"
    )
    _draw_vector_field_frame(
        axes[0],
        data=data,
        frame_index=frame_index,
        log_speed=log_speed,
        speed_norm=speed_norm,
        speed_floor=speed_floor,
        arrow_stride=arrow_stride,
        arrow_color=arrow_color,
        arrow_length_fraction=arrow_length_fraction,
        arrow_width=arrow_width,
        trajectory_line_width=trajectory_line_width,
        decision_band_logit_half_width=decision_band_logit_half_width,
        response_threshold=response_threshold,
        title=f"latent vector field — {epoch_title}",
    )
    _draw_latent_jacobian_frame(
        axes[1],
        data=data,
        frame_index=frame_index,
        norm=jacobian_norm,
        trajectory_line_width=trajectory_line_width,
        title=f"Jacobian spectral abscissa — {epoch_title}",
    )
    _set_latent_figure_legend(
        axes[0].figure,
        data=data,
        frame_index=frame_index,
        include_vector_field=True,
        include_jacobian=True,
        trajectory_line_width=trajectory_line_width,
        decision_band_logit_half_width=decision_band_logit_half_width,
        response_threshold=response_threshold,
    )


def render_latent_jacobian_collection(
    diagnostics_path: str | Path,
    output_dir: str | Path,
    *,
    representative_epochs: Sequence[int],
    representative_labels: Mapping[int, str] | None = None,
    figure_size: Sequence[float],
    dpi: int,
    trajectory_line_width: float = _DEFAULT_TRAJECTORY_LINE_WIDTH,
    spectral_abscissa_limit: float | None = None,
) -> list[Path]:
    """Render representative spectral-abscissa maps on the aligned grid."""
    with np.load(diagnostics_path) as stored:
        if int(stored.get("coordinate_dimension", np.asarray(2))) >= 3:
            return []
    data = _load_vector_field_data(diagnostics_path)
    available = {int(epoch) for epoch in data["epochs"]}
    snapshot_epochs = [int(e) for e in representative_epochs if int(e) in available]

    spectral_abscissa = data["jacobian_spectral_abscissa"]
    norm = _spectral_abscissa_norm(spectral_abscissa, spectral_abscissa_limit)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return [
        plot_latent_jacobian_snapshot(
            diagnostics_path,
            destination / f"epoch-{epoch:06d}.png",
            epoch=epoch,
            representative_label=(representative_labels or {}).get(epoch),
            figure_size=figure_size,
            dpi=dpi,
            trajectory_line_width=trajectory_line_width,
            spectral_abscissa_norm=norm,
        )
        for epoch in snapshot_epochs
    ]


def plot_latent_jacobian_snapshot(
    diagnostics_path: str | Path,
    output_path: str | Path,
    *,
    epoch: int,
    representative_label: str | None = None,
    figure_size: Sequence[float],
    dpi: int,
    trajectory_line_width: float = _DEFAULT_TRAJECTORY_LINE_WIDTH,
    spectral_abscissa_limit: float | None = None,
    spectral_abscissa_norm: Normalize | None = None,
) -> Path:
    """Plot ``max Re(lambda(J_kappa))`` across the aligned kappa plane."""
    data = _load_vector_field_data(diagnostics_path)
    frame_index = _epoch_index(data["epochs"], epoch)
    spectral_abscissa = data["jacobian_spectral_abscissa"]
    norm = spectral_abscissa_norm or _spectral_abscissa_norm(
        spectral_abscissa, spectral_abscissa_limit
    )
    figure, axis = plt.subplots(figsize=tuple(figure_size), constrained_layout=True)
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap="coolwarm"),
        ax=axis,
        label=r"$\max_i\,\mathrm{Re}\,\lambda_i(J_\kappa)$",
    )
    colorbar.ax.tick_params(labelsize=8)
    _draw_latent_jacobian_frame(
        axis,
        data=data,
        frame_index=frame_index,
        norm=norm,
        trajectory_line_width=trajectory_line_width,
        title=(
            "latent Jacobian spectral abscissa — "
            + (
                f"{representative_label} — epoch {int(epoch)}"
                if representative_label
                else f"epoch {int(epoch)}"
            )
        ),
    )
    _set_latent_figure_legend(
        figure,
        data=data,
        frame_index=frame_index,
        include_vector_field=False,
        include_jacobian=True,
        trajectory_line_width=trajectory_line_width,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_latent_vector_field_snapshot(
    diagnostics_path: str | Path,
    output_path: str | Path,
    *,
    epoch: int,
    representative_label: str | None = None,
    speed_floor: float,
    figure_size: Sequence[float],
    dpi: int,
    arrow_stride: int,
    arrow_color: str = "black",
    arrow_length_fraction: float = 0.0075,
    arrow_width: float = 0.0012,
    trajectory_line_width: float = _DEFAULT_TRAJECTORY_LINE_WIDTH,
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
    response_threshold: float = 0.5,
) -> Path:
    """Render one aligned latent vector field as a PNG for the given epoch."""
    data = _load_vector_field_data(diagnostics_path)
    frame_index = _epoch_index(data["epochs"], epoch)
    log_speed = _log_speed(data["speed"], speed_floor)
    speed_norm = Normalize(vmin=float(log_speed.min()), vmax=float(log_speed.max()))
    figure, axis = plt.subplots(figsize=tuple(figure_size), constrained_layout=True)
    colorbar = figure.colorbar(
        ScalarMappable(norm=speed_norm, cmap="viridis"),
        ax=axis,
        label=r"$\log_{10}\,||\tau F_\kappa||_2$",
    )
    colorbar.ax.tick_params(labelsize=8)
    _draw_vector_field_frame(
        axis,
        data=data,
        frame_index=frame_index,
        log_speed=log_speed,
        speed_norm=speed_norm,
        speed_floor=speed_floor,
        arrow_stride=arrow_stride,
        arrow_color=arrow_color,
        arrow_length_fraction=arrow_length_fraction,
        arrow_width=arrow_width,
        trajectory_line_width=trajectory_line_width,
        decision_band_logit_half_width=decision_band_logit_half_width,
        response_threshold=response_threshold,
        title=(
            "autonomous latent vector field — "
            + (
                f"{representative_label} — epoch {int(epoch)}"
                if representative_label
                else f"epoch {int(epoch)}"
            )
        ),
    )
    _set_latent_figure_legend(
        figure,
        data=data,
        frame_index=frame_index,
        include_vector_field=True,
        include_jacobian=False,
        trajectory_line_width=trajectory_line_width,
        decision_band_logit_half_width=decision_band_logit_half_width,
        response_threshold=response_threshold,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def create_latent_vector_field_movie(
    diagnostics_path: str | Path,
    output_path: str | Path,
    *,
    speed_floor: float,
    figure_size: Sequence[float],
    dpi: int,
    fps: int,
    codec: str,
    arrow_stride: int,
    arrow_color: str = "black",
    arrow_length_fraction: float = 0.0075,
    arrow_width: float = 0.0012,
    trajectory_line_width: float = _DEFAULT_TRAJECTORY_LINE_WIDTH,
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
    response_threshold: float = 0.5,
) -> Path:
    """Render aligned autonomous latent vector fields as an H.264 MP4."""
    data = _load_vector_field_data(diagnostics_path)
    log_speed = _log_speed(data["speed"], speed_floor)
    speed_norm = Normalize(vmin=float(log_speed.min()), vmax=float(log_speed.max()))
    figure, axis = plt.subplots(figsize=tuple(figure_size), constrained_layout=True)
    colorbar = figure.colorbar(
        ScalarMappable(norm=speed_norm, cmap="viridis"),
        ax=axis,
        label=r"$\log_{10}\,||\tau F_\kappa||_2$",
    )
    colorbar.ax.tick_params(labelsize=8)
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = FFMpegWriter(
        fps=fps,
        codec=codec,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(figure, path, dpi=dpi):
        for frame_index, epoch in enumerate(data["epochs"]):
            axis.clear()
            _draw_vector_field_frame(
                axis,
                data=data,
                frame_index=frame_index,
                log_speed=log_speed,
                speed_norm=speed_norm,
                speed_floor=speed_floor,
                arrow_stride=arrow_stride,
                arrow_color=arrow_color,
                arrow_length_fraction=arrow_length_fraction,
                arrow_width=arrow_width,
                trajectory_line_width=trajectory_line_width,
                decision_band_logit_half_width=decision_band_logit_half_width,
                response_threshold=response_threshold,
                title=f"autonomous latent vector field — epoch {int(epoch)}",
            )
            _set_latent_figure_legend(
                figure,
                data=data,
                frame_index=frame_index,
                include_vector_field=True,
                include_jacobian=False,
                trajectory_line_width=trajectory_line_width,
                decision_band_logit_half_width=decision_band_logit_half_width,
                response_threshold=response_threshold,
            )
            writer.grab_frame()
    plt.close(figure)
    return path


def _load_vector_field_data(
    diagnostics_path: str | Path,
) -> dict[str, np.ndarray]:
    with np.load(diagnostics_path) as data:
        return {name: data[name] for name in data.files}


def _log_speed(speed: np.ndarray, speed_floor: float) -> np.ndarray:
    return np.log10(np.maximum(speed, speed_floor))


def _robust_positive_log_norm(values: np.ndarray) -> LogNorm:
    finite = np.asarray(values)[
        np.isfinite(values) & (np.asarray(values) > 0)
    ]
    if finite.size == 0:
        return LogNorm(vmin=1.0e-12, vmax=1.0e-4)
    vmin, vmax = np.quantile(finite, [0.02, 0.98])
    vmin = max(float(vmin), np.finfo(float).tiny)
    vmax = float(vmax)
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax, clip=True)


def _spectral_abscissa_norm(
    spectral_abscissa: np.ndarray, configured_limit: float | None
) -> Normalize:
    if configured_limit is not None and configured_limit <= 0:
        raise ValueError("jacobian_spectral_abscissa_limit must be positive or null")
    finite = np.abs(np.asarray(spectral_abscissa)[np.isfinite(spectral_abscissa)])
    limit = float(configured_limit) if configured_limit is not None else (
        float(np.quantile(finite, 0.98)) if finite.size else 1.0
    )
    if not np.isfinite(limit):
        raise ValueError("Jacobian spectral abscissa contains no finite values")
    limit = max(limit, np.finfo(np.float32).eps)
    return Normalize(vmin=-limit, vmax=limit)


def _draw_latent_jacobian_frame(
    axis,
    *,
    data: Mapping[str, np.ndarray],
    frame_index: int,
    norm: Normalize,
    trajectory_line_width: float,
    title: str,
) -> None:
    grid_x = data["grid_x"]
    grid_y = data["grid_y"]
    values = data["jacobian_spectral_abscissa"][frame_index]
    axis.pcolormesh(
        grid_x,
        grid_y,
        values,
        shading="auto",
        cmap="coolwarm",
        norm=norm,
    )
    if float(values.min()) <= 0.0 <= float(values.max()):
        axis.contour(
            grid_x,
            grid_y,
            values,
            levels=[0.0],
            colors=["black"],
            linewidths=0.8,
            linestyles="--",
        )
    _draw_latent_context(
        axis,
        data=data,
        frame_index=frame_index,
        trajectory_line_width=trajectory_line_width,
    )
    axis.set_xlim(float(grid_x.min()), float(grid_x.max()))
    axis.set_ylim(float(grid_y.min()), float(grid_y.max()))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"aligned $\kappa_1$")
    axis.set_ylabel(r"aligned $\kappa_2$")
    axis.set_title(title)


def _draw_vector_field_frame(
    axis,
    *,
    data: Mapping[str, np.ndarray],
    frame_index: int,
    log_speed: np.ndarray,
    speed_norm: Normalize,
    speed_floor: float,
    arrow_stride: int,
    arrow_color: str,
    arrow_length_fraction: float,
    arrow_width: float,
    trajectory_line_width: float,
    decision_band_logit_half_width: float,
    response_threshold: float,
    title: str,
) -> None:
    """Draw one aligned vector-field frame onto an existing axis.

    Dinc decision-band convention: the readout ``y = tanh(z)`` is ambiguous
    inside the band ``|z| <= w`` (output between the two class readouts
    ``tanh(-w)`` and ``tanh(w)``).  Only that band is tinted: the lower half
    ``-w <= z <= 0`` (output < 0, leaning toward -1) red, the upper half
    ``0 <= z <= w`` (output > 0, leaning toward +1) green.  States outside
    the band (``|z| > w``) are clearly committed to ``-1`` or ``+1`` and are
    left untinted so the speed field stays visible.
    """
    grid_x = data["grid_x"]
    grid_y = data["grid_y"]
    axis.pcolormesh(
        grid_x,
        grid_y,
        log_speed[frame_index],
        shading="auto",
        cmap="viridis",
        norm=speed_norm,
    )
    flow_frame = data["flow"][frame_index]
    magnitude = np.linalg.norm(flow_frame, axis=-1, keepdims=True)
    direction = flow_frame / np.maximum(magnitude, speed_floor)
    if arrow_length_fraction <= 0 or arrow_width <= 0:
        raise ValueError("arrow_length_fraction and arrow_width must be positive")
    coordinate_span = max(float(np.ptp(grid_x)), float(np.ptp(grid_y)))
    arrow_length = arrow_length_fraction * coordinate_span
    arrow_vectors = direction * arrow_length
    selection = np.s_[::arrow_stride, ::arrow_stride]
    axis.quiver(
        grid_x[selection],
        grid_y[selection],
        arrow_vectors[selection][..., 0],
        arrow_vectors[selection][..., 1],
        color=arrow_color,
        alpha=0.75,
        pivot="mid",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=arrow_width,
    )
    grid_output_frame = data["grid_output"][frame_index]
    if _is_reproduction(data):
        axis.contour(
            grid_x,
            grid_y,
            grid_output_frame,
            levels=[response_threshold],
            colors=["tab:orange"],
            linewidths=1.0,
            linestyles=":",
        )
    else:
        band_edge = _band_output_edge(decision_band_logit_half_width)
        axis.contourf(
            grid_x, grid_y, grid_output_frame,
            levels=[-1.05, -band_edge, 0.0, band_edge, 1.05],
            colors=[
                (0.0, 0.0, 0.0, 0.0),
                to_rgba(_CLASS_SHORT_COLOR, alpha=_BAND_TINT_ALPHA),
                to_rgba(_CLASS_LONG_COLOR, alpha=_BAND_TINT_ALPHA),
                (0.0, 0.0, 0.0, 0.0),
            ],
        )
        if band_edge > 0.0:
            axis.contour(
                grid_x, grid_y, grid_output_frame, levels=[-band_edge, band_edge],
                colors=["white"], linewidths=0.7, linestyles="-",
            )
        axis.contour(
            grid_x, grid_y, grid_output_frame, levels=[0.0], colors=["white"],
            linewidths=1.0, linestyles="--",
        )
    _draw_latent_context(
        axis,
        data=data,
        frame_index=frame_index,
        trajectory_line_width=trajectory_line_width,
    )
    axis.set_xlim(float(grid_x.min()), float(grid_x.max()))
    axis.set_ylim(float(grid_y.min()), float(grid_y.max()))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"aligned $\kappa_1$")
    axis.set_ylabel(r"aligned $\kappa_2$")
    axis.set_title(title)


def _set_latent_figure_legend(
    figure,
    *,
    data: Mapping[str, np.ndarray],
    frame_index: int,
    include_vector_field: bool,
    include_jacobian: bool,
    trajectory_line_width: float = _DEFAULT_TRAJECTORY_LINE_WIDTH,
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
    response_threshold: float = 0.5,
) -> None:
    """Place one compact shared legend below the latent-data axes."""
    for legend in tuple(figure.legends):
        legend.remove()

    handles: list[Line2D | Patch] = []
    if include_vector_field:
        if _is_reproduction(data):
            handles.append(
                Line2D(
                    [],
                    [],
                    color="tab:orange",
                    linestyle=":",
                    label=f"response threshold y={response_threshold:g}",
                )
            )
        else:
            handles.extend(
                (
                    Patch(
                        facecolor=_CLASS_SHORT_COLOR,
                        alpha=_BAND_TINT_ALPHA,
                        label=(
                            f"−1 decision side "
                            f"(|z|≤{decision_band_logit_half_width:g})"
                        ),
                    ),
                    Patch(
                        facecolor=_CLASS_LONG_COLOR,
                        alpha=_BAND_TINT_ALPHA,
                        label=(
                            f"+1 decision side "
                            f"(|z|≤{decision_band_logit_half_width:g})"
                        ),
                    ),
                )
            )
    if include_jacobian:
        values = data["jacobian_spectral_abscissa"][frame_index]
        if float(values.min()) <= 0.0 <= float(values.max()):
            handles.append(
                Line2D(
                    [],
                    [],
                    color="black",
                    linestyle="--",
                    label=r"$\max\,\mathrm{Re}(\lambda)=0$",
                )
            )
    handles.extend(_speed_minimum_legend_handles(data, frame_index))
    handles.extend(_trajectory_condition_legend_handles(data))
    handles.extend(
        _trajectory_phase_legend_handles(
            data, trajectory_line_width=trajectory_line_width
        )
    )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(4, len(handles)),
        fontsize=7,
        frameon=False,
    )


def _draw_latent_context(
    axis,
    *,
    data: Mapping[str, np.ndarray],
    frame_index: int,
    trajectory_line_width: float,
) -> None:
    """Overlay descriptive speed minima and task trajectories."""
    trajectory_effects = _trajectory_path_effects(trajectory_line_width)
    _draw_speed_minima(axis, data=data, frame_index=frame_index)
    trajectory = data["trajectory"][frame_index]
    target = data["target"]
    valid_mask = data["valid_mask"]
    for trial in range(trajectory.shape[0]):
        length = int(valid_mask[trial].sum())
        points = trajectory[trial, :length]
        color = _trajectory_color(data, target, valid_mask, trial)
        if "trial_s1_step" not in data:
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=trajectory_line_width,
                path_effects=trajectory_effects,
            )
        else:
            s1_step = int(data["trial_s1_step"][trial])
            if s1_step > 0:
                baseline = points[: min(s1_step + 1, length)]
                axis.plot(
                    baseline[:, 0],
                    baseline[:, 1],
                    color=color,
                    linewidth=0.55 * trajectory_line_width,
                    alpha=0.35,
                    path_effects=trajectory_effects,
                )
            for phase, start, stop in _trajectory_phase_ranges(
                data, trial=trial, length=length
            ):
                segment = points[start : stop + 1]
                if segment.shape[0] < 2:
                    continue
                axis.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=color,
                    linewidth=trajectory_line_width,
                    linestyle=_TRAJECTORY_PHASE_LINESTYLES[phase],
                    path_effects=trajectory_effects,
                )
        _draw_cue_driven_segments(
            axis,
            data=data,
            trial=trial,
            points=points,
            length=length,
            trajectory_line_width=trajectory_line_width,
        )
        axis.scatter(
            points[-1, 0],
            points[-1, 1],
            color=color,
            edgecolors="white",
            linewidths=0.45 * trajectory_line_width,
            s=max(8.0, 8.0 * trajectory_line_width**2),
            zorder=3,
        )


def _draw_cue_driven_segments(
    axis,
    *,
    data: Mapping[str, np.ndarray],
    trial: int,
    points: np.ndarray,
    length: int,
    trajectory_line_width: float,
) -> None:
    """Overlay state displacements made while the external cue is nonzero."""
    inputs = data.get("inputs")
    if inputs is None:
        return
    for start, stop in _cue_driven_point_ranges(inputs[trial, :length]):
        segment = points[start:stop]
        if segment.shape[0] < 2:
            continue
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            color=_cue_trajectory_color(data),
            linewidth=_cue_line_width(data, trajectory_line_width),
            linestyle="-",
            path_effects=_cue_path_effects(data, trajectory_line_width),
            zorder=4,
        )


def _cue_driven_point_ranges(inputs: np.ndarray) -> list[tuple[int, int]]:
    """Return point slices spanning transitions generated by nonzero inputs."""
    cue_on = np.any(np.abs(inputs) > 0.0, axis=-1)
    padded = np.pad(cue_on.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    input_starts = np.flatnonzero(changes == 1)
    input_stops = np.flatnonzero(changes == -1)
    return [
        (max(int(start) - 1, 0), int(stop))
        for start, stop in zip(input_starts, input_stops)
    ]


def _trajectory_phase_ranges(
    data: Mapping[str, np.ndarray], *, trial: int, length: int
) -> list[tuple[str, int, int]]:
    boundaries = (
        int(data["trial_s1_step"][trial]),
        int(data["trial_s2_step"][trial]),
        int(data["trial_go_step"][trial]),
        int(data["trial_response_step"][trial]),
        length - 1,
    )
    names = tuple(_TRAJECTORY_PHASE_LINESTYLES)
    return [
        (name, max(0, start), min(stop, length - 1))
        for name, start, stop in zip(names, boundaries[:-1], boundaries[1:])
        if start < length and stop > start
    ]


def _trajectory_color(
    data: Mapping[str, np.ndarray],
    target: np.ndarray,
    valid_mask: np.ndarray,
    trial: int,
):
    if not _is_reproduction(data):
        length = int(valid_mask[trial].sum())
        return _trajectory_class_color(_trial_class(target[trial, :length, 0]))
    values = data.get("trial_interval")
    if values is None or len(values) < 2:
        return "tab:purple"
    span = max(float(np.ptp(values)), 1.0)
    return plt.get_cmap(_REPRODUCTION_TRAJECTORY_CMAP)(
        (float(values[trial]) - float(values.min())) / span
    )


def _trajectory_condition_legend_handles(
    data: Mapping[str, np.ndarray],
) -> list[Line2D]:
    if not _is_reproduction(data):
        return [
            Line2D(
                [], [], color=_TRAJECTORY_SHORT_COLOR, label="short trajectory"
            ),
            Line2D(
                [], [], color=_TRAJECTORY_LONG_COLOR, label="long trajectory"
            ),
        ]
    values = data.get("trial_interval")
    if values is None or len(values) > 6:
        return [Line2D([], [], color="tab:purple", label="trajectory color: T")]
    dummy_target = data["target"]
    valid_mask = data["valid_mask"]
    return [
        Line2D(
            [],
            [],
            color=_trajectory_color(data, dummy_target, valid_mask, trial),
            label=f"T={float(interval):g}",
        )
        for trial, interval in enumerate(values)
    ]


def _trajectory_phase_legend_handles(
    data: Mapping[str, np.ndarray],
    *,
    trajectory_line_width: float,
) -> list[Line2D]:
    post_go_label = "reproduction" if _is_reproduction(data) else "post-Go timing"
    labels = {
        "interval_encoding": "interval encoding",
        "delay": "delay",
        "post_go_timing": post_go_label,
        "response": "response",
    }
    handles = [
        Line2D(
            [],
            [],
            color="black",
            linestyle=linestyle,
            linewidth=trajectory_line_width,
            label=labels[phase],
        )
        for phase, linestyle in _TRAJECTORY_PHASE_LINESTYLES.items()
    ]
    if "inputs" in data:
        handles.append(
            Line2D(
                [],
                [],
                color=_cue_trajectory_color(data),
                linestyle="-",
                linewidth=_cue_line_width(data, trajectory_line_width),
                path_effects=_cue_path_effects(data, trajectory_line_width),
                label=r"cue on ($u\ne0$)",
            )
        )
    return handles


def _is_reproduction(data: Mapping[str, np.ndarray]) -> bool:
    return (
        "task_name" in data
        and str(data["task_name"].item()) == "delayed_interval_reproduction"
    )


def _draw_speed_minima(
    axis, *, data: Mapping[str, np.ndarray], frame_index: int
) -> None:
    """Draw refined minima with their Dinc-style dynamical classification."""
    if "speed_minimum_coordinates" not in data:
        grid_minima = data["speed_minimum_mask"][frame_index]
        axis.scatter(
            data["grid_x"][grid_minima],
            data["grid_y"][grid_minima],
            marker="x",
            color=_UNRESOLVED_MINIMUM_COLOR,
            s=35,
            linewidths=1.5,
            label="_nolegend_",
            zorder=5,
        )
        return

    mask = data["speed_minimum_epoch_index"] == frame_index
    coordinates = data["speed_minimum_coordinates"][mask]
    classifications = data["speed_minimum_classification"][mask]
    attracting = data["speed_minimum_is_attractor"][mask]
    fixed = classifications == "fixed_point"
    styles = (
        (
            fixed & attracting,
            "o",
            _FIXED_POINT_COLOR,
            _FIXED_POINT_COLOR,
            45,
        ),
        (fixed & ~attracting, "o", "none", _FIXED_POINT_COLOR, 45),
        (
            classifications == "slow_point",
            "^",
            _SLOW_POINT_COLOR,
            "white",
            55,
        ),
        (
            classifications == "latent_ghost_candidate",
            "*",
            _GHOST_CANDIDATE_COLOR,
            "white",
            85,
        ),
        (
            classifications == "unresolved",
            "x",
            _UNRESOLVED_MINIMUM_COLOR,
            _UNRESOLVED_MINIMUM_COLOR,
            40,
        ),
    )
    for selected, marker, facecolor, edgecolor, size in styles:
        if not np.any(selected):
            continue
        colors = (
            {"color": facecolor}
            if marker == "x"
            else {"facecolors": facecolor, "edgecolors": edgecolor}
        )
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            marker=marker,
            s=size,
            linewidths=1.1,
            label="_nolegend_",
            zorder=5,
            **colors,
        )


def _speed_minimum_legend_handles(
    data: Mapping[str, np.ndarray], frame_index: int
) -> list[Line2D]:
    if "speed_minimum_coordinates" not in data:
        return [
            Line2D(
                [],
                [],
                marker="x",
                color=_UNRESOLVED_MINIMUM_COLOR,
                linestyle="None",
                label="unrefined grid minimum",
            )
        ]
    mask = data["speed_minimum_epoch_index"] == frame_index
    classifications = data["speed_minimum_classification"][mask]
    attracting = data["speed_minimum_is_attractor"][mask]
    fixed = classifications == "fixed_point"
    handles: list[Line2D] = []
    if np.any(fixed & attracting):
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor=_FIXED_POINT_COLOR,
                markeredgecolor=_FIXED_POINT_COLOR,
                color="none",
                label="F: fixed attractor",
            )
        )
    if np.any(fixed & ~attracting):
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor=_FIXED_POINT_COLOR,
                color="none",
                label="F: unstable fixed point",
            )
        )
    if np.any(classifications == "slow_point"):
        handles.append(
            Line2D(
                [],
                [],
                marker="^",
                markerfacecolor=_SLOW_POINT_COLOR,
                markeredgecolor="white",
                color="none",
                label="S: slow point",
            )
        )
    if np.any(classifications == "latent_ghost_candidate"):
        handles.append(
            Line2D(
                [],
                [],
                marker="*",
                markerfacecolor=_GHOST_CANDIDATE_COLOR,
                markeredgecolor="white",
                color="none",
                markersize=10,
                label="G*: latent ghost candidate",
            )
        )
    if np.any(classifications == "unresolved"):
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                color=_UNRESOLVED_MINIMUM_COLOR,
                linestyle="None",
                label="unresolved minimum",
            )
        )
    return handles


def _read_metrics(path: str | Path) -> list[Mapping[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _epoch_index(epochs: np.ndarray, epoch: int) -> int:
    matches = np.flatnonzero(epochs == epoch)
    if len(matches) != 1:
        raise ValueError(f"Epoch {epoch} is not available exactly once")
    return int(matches[0])
