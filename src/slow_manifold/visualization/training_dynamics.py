"""Dinc-style training and latent-dynamics figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import imageio_ffmpeg
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Decision-side colors shared by every figure: class -1 / short (T < T_c) is
# red, class +1 / long (T > T_c) is green, mirroring the Dinc-style panels.
# The colors also tint the two halves of the decision band (the ambiguous,
# un-committed region |logit| <= w): the lower half (output < 0, leaning -1)
# is red and the upper half (output > 0, leaning +1) is green.
_CLASS_SHORT_COLOR = "tab:red"
_CLASS_LONG_COLOR = "tab:green"
# Default decision-band half-width in readout-logit units (Dinc convention:
# the band spans logits (-1, 1), i.e. sigma(-1)..sigma(1) for a sigmoid
# readout).  States inside the band are "not yet committed"; states outside
# are clearly committed to -1 or +1 and are left untinted (speed field only).
_DECISION_BAND_LOGIT_HALF_WIDTH = 1.0
_BAND_TINT_ALPHA = 0.35


def _band_output_edge(logit_half_width: float) -> float:
    """Readout-output threshold |y| = tanh(w) for logit band half-width w."""
    return float(np.tanh(max(float(logit_half_width), 0.0)))


def _trial_class(target_row: np.ndarray) -> int:
    """Target class of one trial: +1 (long) if any positive target else -1."""
    return 1 if target_row.sum() > 0 else -1


def _class_color(class_value: int) -> str:
    return _CLASS_LONG_COLOR if class_value == 1 else _CLASS_SHORT_COLOR


def plot_training_curves(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    representative_epochs: Sequence[int],
    figure_size: Sequence[float],
) -> Path:
    """Plot loss and gradient norms against epoch on aligned panels."""
    rows = _read_metrics(metrics_path)
    epochs = np.array([int(row["epoch"]) for row in rows])
    train_loss = np.array([float(row["train_loss"]) for row in rows])
    validation_loss = np.array([float(row["validation_loss"]) for row in rows])
    recurrent_gradient = np.array(
        [float(row["recurrent_gradient_norm"]) for row in rows]
    )
    total_gradient = np.array([float(row["total_gradient_norm"]) for row in rows])

    figure, axes = plt.subplots(
        2, 1, figsize=tuple(figure_size), sharex=True, constrained_layout=True
    )
    axes[0].semilogy(epochs, train_loss, label="train loss", color="black")
    axes[0].semilogy(
        epochs, validation_loss, label="validation loss", color="tab:blue"
    )
    axes[0].set_ylabel("masked MSE")
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

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_representative_outputs(
    diagnostics_path: str | Path,
    output_path: str | Path,
    *,
    representative_epochs: Sequence[int],
    dt: float,
    panel_width: float,
    panel_height: float,
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
) -> Path:
    """Compare network and target outputs at selected epochs.

    When the diagnostics carry the input waveforms, each trial gets an extra
    schematic strip on top (Ramesan Fig. 1a style): the S1/S2/Go square-wave
    pulses as black step lines, one band per channel.  The output panels below
    share that trial's time axis.  Runs whose stored diagnostics predate the
    ``inputs`` field fall back to output-only panels.  A translucent band
    marks the readout-logit decision band (Dinc convention).
    """
    with np.load(diagnostics_path) as data:
        epochs = data["epochs"]
        prediction = data["prediction"]
        target = data["target"]
        valid_mask = data["valid_mask"]
        inputs = data["inputs"] if "inputs" in data.files else None

    indices = [_epoch_index(epochs, epoch) for epoch in representative_epochs]
    trials = target.shape[0]
    n_columns = len(indices)
    if inputs is not None:
        return _plot_representative_outputs_with_inputs(
            diagnostics_path=diagnostics_path,
            output_path=output_path,
            inputs=inputs,
            prediction=prediction,
            target=target,
            valid_mask=valid_mask,
            representative_epochs=representative_epochs,
            indices=indices,
            trials=trials,
            n_columns=n_columns,
            dt=dt,
            panel_width=panel_width,
            panel_height=panel_height,
            decision_band_logit_half_width=decision_band_logit_half_width,
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
                axis.set_title(f"epoch {epoch}")
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
    figure.savefig(path, dpi=150)
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
    indices: Sequence[int],
    trials: int,
    n_columns: int,
    dt: float,
    panel_width: float,
    panel_height: float,
    decision_band_logit_half_width: float,
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
                axis.set_title(f"epoch {epoch}")
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
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _draw_input_schematic(
    axis,
    *,
    inputs: np.ndarray,
    time: np.ndarray,
) -> None:
    """Draw S1/S2/Go square-wave pulses as black step lines, one band each."""
    channel_names = ("S1", "S2", "Go")
    amplitude = float(np.max(np.abs(inputs))) if inputs.size else 1.0
    amplitude = max(amplitude, 1e-9)
    band = 1.0
    for channel, name in enumerate(channel_names):
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
    speed_floor: float,
    figure_size: Sequence[float],
    dpi: int,
    fps: int,
    codec: str,
    arrow_stride: int,
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
) -> dict[str, Path]:
    """Isolated folder with one snapshot per representative epoch + the MP4."""
    with np.load(diagnostics_path) as data:
        available = {int(epoch) for epoch in data["epochs"]}
        final_epoch = int(data["epochs"][-1])
    snapshot_epochs = [int(e) for e in representative_epochs if int(e) in available]
    if final_epoch not in snapshot_epochs:
        snapshot_epochs.append(final_epoch)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshots: list[Path] = []
    for epoch in snapshot_epochs:
        path = plot_latent_vector_field_snapshot(
            diagnostics_path,
            destination / f"epoch-{epoch:06d}.png",
            epoch=epoch,
            speed_floor=speed_floor,
            figure_size=figure_size,
            dpi=dpi,
            arrow_stride=arrow_stride,
            decision_band_logit_half_width=decision_band_logit_half_width,
        )
        snapshots.append(path)
    movie = create_latent_vector_field_movie(
        diagnostics_path,
        destination / "latent_vector_field.mp4",
        speed_floor=speed_floor,
        figure_size=figure_size,
        dpi=dpi,
        fps=fps,
        codec=codec,
        arrow_stride=arrow_stride,
        decision_band_logit_half_width=decision_band_logit_half_width,
    )
    return {"snapshots": snapshots, "movie": movie}


def plot_latent_vector_field_snapshot(
    diagnostics_path: str | Path,
    output_path: str | Path,
    *,
    epoch: int,
    speed_floor: float,
    figure_size: Sequence[float],
    dpi: int,
    arrow_stride: int,
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
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
        decision_band_logit_half_width=decision_band_logit_half_width,
        title=f"autonomous latent vector field — epoch {int(epoch)}",
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
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
    decision_band_logit_half_width: float = _DECISION_BAND_LOGIT_HALF_WIDTH,
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
                decision_band_logit_half_width=decision_band_logit_half_width,
                title=f"autonomous latent vector field — epoch {int(epoch)}",
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


def _draw_vector_field_frame(
    axis,
    *,
    data: Mapping[str, np.ndarray],
    frame_index: int,
    log_speed: np.ndarray,
    speed_norm: Normalize,
    speed_floor: float,
    arrow_stride: int,
    decision_band_logit_half_width: float,
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
    selection = np.s_[::arrow_stride, ::arrow_stride]
    axis.quiver(
        grid_x[selection],
        grid_y[selection],
        direction[selection][..., 0],
        direction[selection][..., 1],
        color="white",
        alpha=0.8,
        pivot="mid",
    )
    band_edge = _band_output_edge(decision_band_logit_half_width)
    grid_output_frame = data["grid_output"][frame_index]
    axis.contourf(
        grid_x,
        grid_y,
        grid_output_frame,
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
            grid_x,
            grid_y,
            grid_output_frame,
            levels=[-band_edge, band_edge],
            colors=["white"],
            linewidths=0.7,
            linestyles="-",
        )
    axis.contour(
        grid_x,
        grid_y,
        grid_output_frame,
        levels=[0.0],
        colors=["white"],
        linewidths=1.0,
        linestyles="--",
    )
    minima = data["speed_minimum_mask"][frame_index]
    axis.scatter(
        grid_x[minima],
        grid_y[minima],
        marker="x",
        color="magenta",
        s=35,
        linewidths=1.5,
        label="_nolegend_",
        zorder=4,
    )
    trajectory = data["trajectory"][frame_index]
    target = data["target"]
    valid_mask = data["valid_mask"]
    for trial in range(trajectory.shape[0]):
        length = int(valid_mask[trial].sum())
        points = trajectory[trial, :length]
        class_value = _trial_class(target[trial, :length, 0])
        color = _class_color(class_value)
        axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5)
        axis.scatter(points[-1, 0], points[-1, 1], color=color, s=15, zorder=3)
    axis.set_xlim(float(grid_x.min()), float(grid_x.max()))
    axis.set_ylim(float(grid_y.min()), float(grid_y.max()))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"aligned $\kappa_1$")
    axis.set_ylabel(r"aligned $\kappa_2$")
    axis.set_title(title)
    band_annotation = (
        f"$|z|\\leq{decision_band_logit_half_width:g}$ "
        f"($|y|\\leq{band_edge:.3g}$)"
    )
    region_legend = [
        Patch(
            facecolor=_CLASS_SHORT_COLOR,
            alpha=_BAND_TINT_ALPHA,
            label=(
                f"decision band, $-1$ side\n(uncommitted, leaning $-1$; "
                f"{band_annotation})"
            ),
        ),
        Patch(
            facecolor=_CLASS_LONG_COLOR,
            alpha=_BAND_TINT_ALPHA,
            label=(
                f"decision band, $+1$ side\n(uncommitted, leaning $+1$; "
                f"{band_annotation})"
            ),
        ),
        Line2D(
            [],
            [],
            marker="x",
            color="magenta",
            linestyle="None",
            label="grid speed minimum",
        ),
    ]
    axis.legend(handles=region_legend, loc="upper right", fontsize=7)


def _read_metrics(path: str | Path) -> list[Mapping[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _epoch_index(epochs: np.ndarray, epoch: int) -> int:
    matches = np.flatnonzero(epochs == epoch)
    if len(matches) != 1:
        raise ValueError(f"Epoch {epoch} is not available exactly once")
    return int(matches[0])
