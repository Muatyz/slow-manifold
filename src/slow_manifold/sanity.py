"""Reusable Phase 0 task-sanity workflow."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from slow_manifold.config import ResolvedExperiment, dump_yaml, resolve_experiment
from slow_manifold.tasks import create_task
from slow_manifold.utils import collect_runtime_metadata, make_rng
from slow_manifold.visualization import plot_task_batch


def create_task_sanity_run(
    experiment_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    split: str = "train",
    batch_size: int | None = None,
) -> Path:
    """Generate a traceable task batch and its sanity-check figure."""
    experiment = resolve_experiment(experiment_path)
    _require_components(experiment, "task", "model", "visualization")

    model_config = experiment.components["model"]
    if "dt" not in model_config:
        raise ValueError("Model component must define dt")
    visualization_config = experiment.components["visualization"]
    requested_batch_size = (
        int(batch_size)
        if batch_size is not None
        else int(visualization_config["batch_size"])
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

    rng_namespace = f"task:{split}"
    rng = make_rng(experiment.seed, rng_namespace)
    task = create_task(
        experiment.components["task"], dt=float(model_config["dt"]), split=split
    )
    batch = task.generate_batch(requested_batch_size, rng)
    expected_inputs = int(model_config.get("input_size", batch.inputs.shape[-1]))
    if batch.inputs.shape[-1] != expected_inputs:
        raise ValueError(
            "Task input channels do not match model input_size: "
            f"{batch.inputs.shape[-1]} != {expected_inputs}"
        )

    resolved = experiment.as_dict()
    resolved["run"] = {
        "kind": "task_sanity",
        "split": split,
        "condition_label": experiment.condition_label,
        "condition_fingerprint": experiment.condition_fingerprint,
        "output_dir": str(run_dir),
    }
    dump_yaml(resolved, run_dir / "config.yaml")
    dump_yaml(
        collect_runtime_metadata(
            experiment_name=experiment.name,
            seed=experiment.seed,
            rng_streams={"task": rng_namespace},
            extra={
                "condition_label": experiment.condition_label,
                "condition_fingerprint": experiment.condition_fingerprint,
            },
        ),
        run_dir / "metadata.yaml",
    )
    dump_yaml(
        {"trials": [asdict(item) for item in batch.metadata]},
        run_dir / "trial_metadata.yaml",
    )
    np.savez_compressed(
        run_dir / "batch.npz",
        inputs=batch.inputs,
        target=batch.target,
        loss_mask=batch.loss_mask,
        valid_mask=batch.valid_mask,
    )
    plot_task_batch(
        batch,
        run_dir / "figures" / "task_trials.png",
        display_trials=int(visualization_config["display_trials"]),
        figure_size=visualization_config["figure_size"],
    )
    return run_dir


def _require_components(experiment: ResolvedExperiment, *names: str) -> None:
    missing = set(names) - set(experiment.components)
    if missing:
        raise ValueError(f"Missing experiment components: {', '.join(sorted(missing))}")
