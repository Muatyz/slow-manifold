import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from slow_manifold.analysis import (
    CheckpointSelectionError,
    RepresentativeSelectionConfig,
    select_representative_checkpoints,
)
from slow_manifold.workflows._stage_config import analysis_fingerprint
from slow_manifold.workflows.visualize_run import (
    _stored_latent_analysis_fingerprint,
)


def _write_metrics(path: Path, losses: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("epoch", "validation_loss")
        )
        writer.writeheader()
        for epoch, loss in enumerate(losses):
            writer.writerow({"epoch": epoch, "validation_loss": float(loss)})


def test_auto_selection_labels_an_abrupt_transition_and_writes_manifest(
    tmp_path: Path,
) -> None:
    losses = np.concatenate(
        (
            np.ones(40),
            np.linspace(1.0, 0.1, 5),
            np.full(56, 0.1),
        )
    )
    metrics = tmp_path / "metrics.csv"
    manifest = tmp_path / "checkpoint_selection.yaml"
    _write_metrics(metrics, losses)

    result = select_representative_checkpoints(
        metrics_path=metrics,
        checkpoint_epochs=range(len(losses)),
        config=RepresentativeSelectionConfig(),
        output_path=manifest,
        source="test:auto",
    )

    assert result.transition_classification == "abrupt"
    assert result.terminal_regime == "quasi_plateau"
    assert result.epochs[0] == 0
    assert result.selected[0].role == "initialization"
    roles = {item.role for item in result.selected}
    assert roles >= {"post_transition", "mature"}
    assert roles & {"pre_transition", "transition"}
    record = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert record["selection_mode"] == "auto"
    assert record["schema_version"] == 2
    assert record["transition"]["classification"] == "abrupt"
    assert record["selected"][-1]["role"] == "mature"
    assert record["selected"][-1]["selection_method"] == (
        "best_raw_in_best_smoothed_window"
    )
    assert len(record["metrics_sha256"]) == 64


def test_auto_selection_calls_a_smooth_trace_gradual(tmp_path: Path) -> None:
    losses = np.exp(np.linspace(0.0, -2.0, 101))
    metrics = tmp_path / "metrics.csv"
    _write_metrics(metrics, losses)

    result = select_representative_checkpoints(
        metrics_path=metrics,
        checkpoint_epochs=range(len(losses)),
        config=RepresentativeSelectionConfig(),
        output_path=tmp_path / "selection.yaml",
        source="test:auto",
    )

    assert result.transition_classification == "gradual"
    assert any(item.role == "progress_50" for item in result.selected)
    assert result.selected[-1].role == "mature"


def test_mature_uses_a_good_raw_checkpoint_from_the_best_smoothed_window(
    tmp_path: Path,
) -> None:
    losses = np.concatenate(
        (
            np.ones(40),
            np.linspace(1.0, 0.2, 5),
            np.full(55, 0.2),
        )
    )
    losses[-1] = 0.5
    metrics = tmp_path / "metrics.csv"
    manifest = tmp_path / "selection.yaml"
    _write_metrics(metrics, losses)

    result = select_representative_checkpoints(
        metrics_path=metrics,
        checkpoint_epochs=range(len(losses)),
        config=RepresentativeSelectionConfig(),
        output_path=manifest,
        source="test:auto",
    )

    mature = result.selected[-1]
    assert mature.role == "mature"
    assert mature.anchor_epoch == len(losses) - 1
    assert mature.epoch == len(losses) - 2
    assert mature.metric_value == pytest.approx(0.2)
    assert mature.selection_method == "best_raw_in_best_smoothed_window"
    assert (mature.window_epoch_min, mature.window_epoch_max) == (97, 99)


def test_manual_selection_requires_stored_checkpoints(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    _write_metrics(metrics, np.linspace(1.0, 0.5, 11))
    config = RepresentativeSelectionConfig.from_mapping(
        {"mode": "manual", "manual_epochs": [0, 7]}
    )

    with pytest.raises(CheckpointSelectionError, match="no stored checkpoint"):
        select_representative_checkpoints(
            metrics_path=metrics,
            checkpoint_epochs=[0, 5, 10],
            config=config,
            output_path=tmp_path / "selection.yaml",
            source="test:manual",
        )


def test_selection_skips_epochs_without_validation_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("epoch", "train_loss", "validation_loss")
        )
        writer.writeheader()
        for epoch in range(5):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": 1.0 - 0.1 * epoch,
                    "validation_loss": (
                        1.0 - 0.1 * epoch if epoch % 2 == 0 else None
                    ),
                }
            )

    result = select_representative_checkpoints(
        metrics_path=metrics,
        checkpoint_epochs=[0, 2, 4],
        config=RepresentativeSelectionConfig.from_mapping(
            {"mode": "manual", "manual_epochs": [0, 4]}
        ),
        output_path=tmp_path / "selection.yaml",
        source="test:sparse-validation",
    )

    assert result.epochs == (0, 4)


def test_selection_rules_do_not_change_latent_analysis_fingerprint() -> None:
    common = {
        "name": "analysis",
        "grid_points": 41,
        "representative_selection": {"mode": "auto", "max_epochs": 5},
    }
    manual = {
        **common,
        "representative_selection": {
            "mode": "manual",
            "manual_epochs": [0, 100],
        },
    }

    first = analysis_fingerprint(
        task={"name": "task"},
        model={"name": "model"},
        analysis=common,
        checkpoint_epochs=[0, 100],
    )
    second = analysis_fingerprint(
        task={"name": "task"},
        model={"name": "model"},
        analysis=manual,
        checkpoint_epochs=[0, 100],
    )

    assert first == second


def test_high_rank_slow_point_epochs_change_analysis_fingerprint() -> None:
    common = {
        "name": "analysis",
        "trajectory_slow_point_search": {"enabled": True, "seed_count": 32},
    }
    first = analysis_fingerprint(
        task={"name": "task"},
        model={"name": "model", "rank": 3},
        analysis=common,
        checkpoint_epochs=[0, 100],
        slow_point_epochs=[0],
    )
    second = analysis_fingerprint(
        task={"name": "task"},
        model={"name": "model", "rank": 3},
        analysis=common,
        checkpoint_epochs=[0, 100],
        slow_point_epochs=[100],
    )

    assert first != second


def test_old_stage_fingerprint_is_canonicalized_without_fixed_epochs(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "config.yaml"
    stage.write_text(
        yaml.safe_dump(
            {
                "kind": "analysis",
                "fingerprint": "legacy-value",
                "checkpoint_epochs": [0, 100],
                "config": {
                    "name": "analysis",
                    "grid_points": 41,
                    "representative_epochs": [0, 100],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    canonical = _stored_latent_analysis_fingerprint(
        stage,
        task={"name": "task"},
        model={"name": "model"},
    )
    desired = analysis_fingerprint(
        task={"name": "task"},
        model={"name": "model"},
        analysis={
            "name": "analysis",
            "grid_points": 41,
            "representative_selection": {"mode": "auto"},
        },
        checkpoint_epochs=[0, 100],
    )

    assert canonical == desired
