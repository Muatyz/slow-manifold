import csv
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from slow_manifold.config import resolve_experiment
from slow_manifold.models import LowRankCTRNN, LowRankCTRNNConfig
from slow_manifold.visualization import resolve_dpi_settings, resolve_render_movies
from slow_manifold.workflows import run_low_rank_training


ROOT = Path(__file__).resolve().parents[1]


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def test_snapshot_and_movie_dpi_are_independent_with_legacy_fallback() -> None:
    assert resolve_dpi_settings(
        {"snapshot_dpi": 240, "movie_dpi": 120}
    ) == (240, 120)
    assert resolve_dpi_settings({"vector_field_dpi": 72}) == (72, 72)
    with pytest.raises(ValueError, match="must be positive"):
        resolve_dpi_settings({"snapshot_dpi": 0, "movie_dpi": 120})


def test_movie_switch_defaults_to_historical_behavior_and_requires_boolean() -> None:
    assert resolve_render_movies({}) is True
    assert resolve_render_movies({"render_movies": False}) is False
    with pytest.raises(ValueError, match="must be a boolean"):
        resolve_render_movies({"render_movies": "false"})


def test_exact_rank5_latent_jacobian_matches_autograd() -> None:
    mapping = resolve_experiment(
        ROOT / "experiments" / "phase2_rank5_smoke.yaml"
    ).components["model"]
    config = LowRankCTRNNConfig.from_mapping(mapping)
    model = LowRankCTRNN(config, generator=torch.Generator().manual_seed(7))
    kappa = torch.randn(4, config.rank)
    inputs = torch.randn(4, config.input_size)

    analytic = model.latent_jacobian(kappa, inputs)
    automatic = torch.stack(
        [
            torch.autograd.functional.jacobian(
                lambda point: model.latent_flow(
                    point.unsqueeze(0), inputs[index : index + 1]
                ).squeeze(0),
                kappa[index],
            )
            for index in range(kappa.shape[0])
        ]
    )

    torch.testing.assert_close(analytic, automatic, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize(
    ("rank", "recipe_name", "coordinate_kind", "task_name", "trial_intervals"),
    [
        (
            3,
            "phase2_rank3_smoke.yaml",
            "exact_kappa",
            "delayed_interval_categorization",
            [3.0, 4.5, 5.5, 7.0],
        ),
        (
            5,
            "phase2_rank5_smoke.yaml",
            "trajectory_pca",
            "delayed_interval_categorization",
            [3.0, 4.5, 5.5, 7.0],
        ),
        (
            3,
            "phase2_rank3_IR_smoke.yaml",
            "exact_kappa",
            "delayed_interval_reproduction",
            [30.0, 50.0, 75.0, 99.0],
        ),
        (
            5,
            "phase2_rank5_IR_smoke.yaml",
            "trajectory_pca",
            "delayed_interval_reproduction",
            [30.0, 50.0, 75.0, 99.0],
        ),
    ],
)
def test_high_rank_smoke_pipeline(
    tmp_path: Path,
    rank: int,
    recipe_name: str,
    coordinate_kind: str,
    task_name: str,
    trial_intervals: list[float],
) -> None:
    run_dir = run_low_rank_training(
        ROOT / "experiments" / recipe_name,
        tmp_path / f"rank-{rank}",
    )

    with np.load(run_dir / "diagnostics" / "latent_dynamics.npz") as data:
        assert int(data["latent_rank"]) == rank
        assert int(data["coordinate_dimension"]) == 3
        assert data["coordinate_kind"].item() == coordinate_kind
        assert data["task_name"].item() == task_name
        np.testing.assert_allclose(data["trial_interval"], trial_intervals)
        assert data["trajectory"].shape[-1] == 3
        assert data["trajectory_latent"].shape[-1] == rank
        assert data["neighborhood_latent"].shape[-1] == rank
        assert data["neighborhood_coordinates"].shape[-1] == 3
        assert data["neighborhood_jacobian_eigenvalues"].shape[-1] == rank
        assert data["neighborhood_low_q_mask"].dtype == np.bool_
        assert data["neighborhood_near_zero_mask"].dtype == np.bool_
        assert data["projection_components"].shape[-2:] == (rank, 3)
        for components in data["projection_components"]:
            np.testing.assert_allclose(
                components.T @ components,
                np.eye(3),
                rtol=1.0e-5,
                atol=1.0e-6,
            )

    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert f"singular_value_{rank}" in rows[-1]
    dynamics_snapshot = (
        run_dir / "figures" / "latent_dynamics" / "epoch-000001.png"
    )
    assert dynamics_snapshot.is_file()
    assert not (
        run_dir / "figures" / "latent_dynamics" / "latent_dynamics.mp4"
    ).exists()
    width, height = _png_dimensions(dynamics_snapshot)
    assert width >= 2500
    assert height >= 800
    curve_width, curve_height = _png_dimensions(
        run_dir / "figures" / "loss_and_gradient.png"
    )
    assert curve_width >= 1000
    assert curve_height >= 700

    with (run_dir / "figures" / "config.yaml").open(encoding="utf-8") as stream:
        figure_stage = yaml.safe_load(stream)
    assert figure_stage["config"]["latent_dynamics_3d_figure_size"] == [15.0, 5.0]
    assert figure_stage["config"]["snapshot_dpi"] == 200
    assert figure_stage["config"]["movie_dpi"] == 100
    assert figure_stage["config"]["render_movies"] is False
