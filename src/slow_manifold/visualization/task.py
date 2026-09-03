"""Visual sanity checks for behavioral task batches."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

from slow_manifold.tasks import TaskBatch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_task_batch(
    batch: TaskBatch,
    output_path: str | Path,
    *,
    display_trials: int = 8,
    figure_size: Sequence[float] = (12.0, 14.0),
) -> Path:
    """Plot inputs, targets, and loss masks for representative trials."""
    count = min(display_trials, len(batch.metadata))
    if count <= 0:
        raise ValueError("display_trials must be positive")

    figure, axes = plt.subplots(count, 1, figsize=tuple(figure_size), squeeze=False)
    colors = ("tab:red", "tab:orange", "tab:green")
    for index in range(count):
        axis = axes[index, 0]
        metadata = batch.metadata[index]
        length = metadata.trial_steps
        time = np.arange(length) * batch.dt
        for channel, (name, color) in enumerate(zip(("S1", "S2", "Go"), colors)):
            axis.step(
                time,
                batch.inputs[index, :length, channel],
                where="post",
                color=color,
                label=name,
            )
        axis.step(
            time,
            batch.target[index, :length, 0],
            where="post",
            color="black",
            label="target",
        )
        axis.step(
            time,
            batch.loss_mask[index, :length, 0],
            where="post",
            color="tab:blue",
            linestyle="--",
            alpha=0.7,
            label="loss mask",
        )
        axis.axvline(metadata.response_step * batch.dt, color="0.5", linestyle=":")
        axis.set_ylabel(f"trial {index}\nclass {metadata.class_label:+d}")
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(loc="upper right", ncol=5)
    axes[-1, 0].set_xlabel("time")
    figure.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
