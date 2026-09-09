import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from slow_manifold.config import resolve_experiment
from slow_manifold.models import Rank2CTRNN, Rank2CTRNNConfig
from slow_manifold.training.checkpoint import load_checkpoint
from slow_manifold.workflows import rerun_rank2_visualization, run_rank2_training

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "tests" / "fixtures" / "phase1_smoke.yaml"
REPRODUCTION_SMOKE = ROOT / "experiments" / "phase1_rank2_IR_smoke.yaml"


def test_short_training_run_writes_reproducible_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    expected_run_dir = resolve_experiment(SMOKE).default_run_dir()
    run_dir = run_rank2_training(SMOKE)
    assert run_dir == expected_run_dir

    expected = (
        "config.yaml",
        "metadata.yaml",
        "run.log",
        "metrics.csv",
        "status.yaml",
        "checkpoints/initial.pt",
        "checkpoints/best.pt",
        "checkpoints/final.pt",
        "checkpoints/epoch-000000.pt",
        "checkpoints/epoch-000001.pt",
        "diagnostics/latent_dynamics.npz",
        "diagnostics/latent_dynamics.yaml",
        "figures/loss_and_gradient.png",
        "figures/representative_outputs.png",
        "figures/latent_vector_field/latent_vector_field.mp4",
        "figures/latent_vector_field/epoch-000000.png",
        "figures/latent_vector_field/epoch-000001.png",
        "figures/latent_jacobian/epoch-000000.png",
        "figures/latent_jacobian/epoch-000001.png",
        "figures/latent_dynamics/latent_dynamics.mp4",
        "figures/latent_dynamics/epoch-000000.png",
        "figures/latent_dynamics/epoch-000001.png",
    )
    for relative_path in expected:
        assert (run_dir / relative_path).is_file(), relative_path

    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [0, 1]
    assert all(float(row["recurrent_gradient_norm"]) >= 0 for row in rows)
    for field in (
        "learning_rate",
        "m_norm",
        "n_norm",
        "w_norm",
        "singular_value_1",
        "singular_value_2",
        "epoch_seconds",
        "elapsed_seconds",
    ):
        assert all(np.isfinite(float(row[field])) for row in rows)
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    assert elapsed == sorted(elapsed)

    with (run_dir / "status.yaml").open(encoding="utf-8") as stream:
        status = yaml.safe_load(stream)
    assert status["status"] == "complete"
    assert status["final_epoch"] == 1
    assert "best_epoch" in status
    assert "started_at_utc" in status
    assert "finished_at_utc" in status
    assert status["elapsed_seconds"] >= 0

    experiment = resolve_experiment(SMOKE)
    with (run_dir / "config.yaml").open(encoding="utf-8") as stream:
        saved_config = yaml.safe_load(stream)
    assert saved_config["run"]["condition_label"] == "lr-1e-3"
    assert (
        saved_config["run"]["condition_fingerprint"]
        == experiment.condition_fingerprint
    )
    with (run_dir / "metadata.yaml").open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    assert metadata["condition_fingerprint"] == experiment.condition_fingerprint

    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    for expected_fragment in (
        "run started",
        "configuration Task: Interval Categorization",
        "configuration Model: Vanilla RNN | N=8",
        "configuration Train: optimizer=adam | learning rate=0.001",
        "training started",
        "step=1",
        "training complete",
        "latent analysis complete",
        "vector field artifacts written",
        "latent Jacobian artifacts written",
        "combined latent dynamics artifacts written",
        "run complete",
    ):
        assert expected_fragment in run_log, expected_fragment

    payload = load_checkpoint(run_dir / "checkpoints" / "final.pt")
    assert payload["epoch"] == 1
    assert "resolved_config" in payload
    assert "task_train" in payload["rng_state"]

    with np.load(run_dir / "diagnostics" / "latent_dynamics.npz") as data:
        assert data["epochs"].tolist() == [0, 1]
        assert data["flow"].shape == (2, 7, 7, 2)
        assert data["speed_minimum_mask"].shape == (2, 7, 7)
        assert data["jacobian_eigenvalues"].shape == (2, 7, 7, 2)
        assert data["jacobian_spectral_abscissa"].shape == (2, 7, 7)
        assert np.isfinite(data["jacobian_eigenvalues"]).all()
        np.testing.assert_allclose(
            data["jacobian_spectral_abscissa"],
            data["jacobian_eigenvalues"].real.max(axis=-1),
        )
        candidate_count = data["speed_minimum_coordinates"].shape[0]
        assert data["speed_minimum_epoch_index"].shape == (candidate_count,)
        assert data["speed_minimum_eigenvalues"].shape == (candidate_count, 2)
        assert data["speed_minimum_q_hessian_eigenvalues"].shape == (
            candidate_count,
            2,
        )
        assert set(data["speed_minimum_classification"].tolist()) <= {
            "fixed_point",
            "slow_point",
            "latent_ghost_candidate",
            "unresolved",
        }
        assert data["basis"].shape == (2, 8, 2)
        for field in (
            "trial_s1_step",
            "trial_s2_step",
            "trial_go_step",
            "trial_response_step",
            "trial_steps",
        ):
            assert data[field].shape == (2,)
        assert np.all(data["trial_s1_step"] < data["trial_s2_step"])
        assert np.all(data["trial_s2_step"] < data["trial_go_step"])
        assert np.all(data["trial_go_step"] < data["trial_response_step"])
        final_basis = torch.as_tensor(data["basis"][-1])
        torch.testing.assert_close(final_basis.T @ final_basis, torch.eye(2))
    with (run_dir / "diagnostics" / "latent_dynamics.yaml").open(
        encoding="utf-8"
    ) as stream:
        diagnostics_metadata = yaml.safe_load(stream)
    assert diagnostics_metadata["grid_points"] == 7

    model_config = Rank2CTRNNConfig.from_mapping(experiment.components["model"])
    restored = Rank2CTRNN(model_config)
    restored.load_state_dict(payload["model_state"])
    singular_values = torch.linalg.svdvals(restored.recurrent_matrix)[:2]
    np.testing.assert_allclose(
        singular_values.detach().numpy(),
        [float(rows[-1]["singular_value_1"]), float(rows[-1]["singular_value_2"])],
        rtol=1e-5,
    )
    torch.testing.assert_close(
        restored.recurrent_matrix @ final_basis @ final_basis.T,
        restored.recurrent_matrix,
        rtol=1e-4,
        atol=1e-5,
    )
    outputs, _, _ = restored.rollout(torch.zeros(2, 5, 3))
    assert torch.isfinite(outputs).all()

    first_rng = np.random.default_rng()
    first_rng.bit_generator.state = payload["rng_state"]["task_train"]
    second_payload = load_checkpoint(run_dir / "checkpoints" / "final.pt")
    second_rng = np.random.default_rng()
    second_rng.bit_generator.state = second_payload["rng_state"]["task_train"]
    np.testing.assert_array_equal(first_rng.random(8), second_rng.random(8))


def test_training_run_refuses_to_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        run_rank2_training(SMOKE, output_dir)


def test_reproduction_training_pipeline_reports_timing_metrics(tmp_path: Path) -> None:
    run_dir = run_rank2_training(REPRODUCTION_SMOKE, tmp_path / "reproduction")
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for field in (
        "validation_timing_mae",
        "validation_timing_rmse",
        "validation_timing_bias",
        "validation_premature_rate",
        "validation_no_response_rate",
    ):
        assert field in rows[-1]
    with np.load(run_dir / "diagnostics" / "latent_dynamics.npz") as data:
        assert data["task_name"].item() == "delayed_interval_reproduction"
        assert data["trajectory"].shape[1] == 3
        assert data["trajectory"].shape[-1] == 2
    assert (run_dir / "figures/representative_outputs.png").is_file()
    assert (run_dir / "figures/latent_vector_field/latent_vector_field.mp4").is_file()
    assert (run_dir / "figures/latent_dynamics/latent_dynamics.mp4").is_file()


@pytest.mark.parametrize(
    ("exception", "expected_status"),
    [
        (RuntimeError("synthetic design error"), "failed"),
        (KeyboardInterrupt("synthetic interrupt"), "interrupted"),
    ],
)
def test_run_failure_records_status_and_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: BaseException,
    expected_status: str,
) -> None:
    """A crashed run must say why it stopped and keep the traceback in the log."""
    monkeypatch.chdir(tmp_path)

    import slow_manifold.workflows.rank2_training as workflow

    def _explode(*args: object, **kwargs: object) -> None:
        raise exception

    monkeypatch.setattr(workflow, "train_model", _explode)
    with pytest.raises(type(exception), match="synthetic"):
        workflow.run_rank2_training(SMOKE)

    run_dir = resolve_experiment(SMOKE).default_run_dir()
    with (run_dir / "status.yaml").open(encoding="utf-8") as stream:
        status = yaml.safe_load(stream)
    assert status["status"] == expected_status
    assert status["error"]["type"] == type(exception).__name__
    assert "started_at_utc" in status
    assert "finished_at_utc" in status
    assert status["elapsed_seconds"] >= 0

    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "Traceback" in run_log
    assert "synthetic" in run_log


def test_visualize_rerun_from_existing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-render figures from a stored run without retraining."""
    monkeypatch.chdir(tmp_path)
    run_dir = run_rank2_training(SMOKE)
    original_log = (run_dir / "run.log").read_text(encoding="utf-8")

    import slow_manifold.workflows.visualize_run as visualize_workflow

    def _must_not_recompute(*args: object, **kwargs: object) -> None:
        raise AssertionError("visualization-only rerun must reuse diagnostics")

    monkeypatch.setattr(visualize_workflow, "analyze_checkpoints", _must_not_recompute)
    figures_dir = rerun_rank2_visualization(run_dir, arrow_stride=1)

    assert figures_dir == run_dir / "figures"
    for relative_path in (
        "loss_and_gradient.png",
        "representative_outputs.png",
        "latent_vector_field/latent_vector_field.mp4",
        "latent_vector_field/epoch-000000.png",
        "latent_vector_field/epoch-000001.png",
        "latent_jacobian/epoch-000000.png",
        "latent_jacobian/epoch-000001.png",
        "latent_dynamics/latent_dynamics.mp4",
        "latent_dynamics/epoch-000000.png",
        "latent_dynamics/epoch-000001.png",
    ):
        assert (figures_dir / relative_path).is_file(), relative_path
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "visualization rerun started" in log
    assert "visualization rerun complete" in log
    assert log.count("visualization rerun started") == 1
    # no retraining happened: training events appear exactly once
    assert log.count("training started") == 1
    # visualization-only overrides are recorded but never touch the diagnostics
    with (run_dir / "diagnostics" / "visualize_rerun.yaml").open(
        encoding="utf-8"
    ) as stream:
        marker = yaml.safe_load(stream)
    assert marker["overrides"] == {"arrow_stride": 1}
    assert original_log.count("visualization rerun") == 0


def test_visualize_rerun_upgrades_diagnostics_without_jacobian(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy runs gain Jacobian fields from their stored checkpoints."""
    monkeypatch.chdir(tmp_path)
    run_dir = run_rank2_training(SMOKE)
    data_path = run_dir / "diagnostics" / "latent_dynamics.npz"
    with np.load(data_path) as data:
        legacy = {
            name: data[name]
            for name in data.files
            if name
            not in {"jacobian_eigenvalues", "jacobian_spectral_abscissa"}
        }
    np.savez_compressed(data_path, **legacy)

    rerun_rank2_visualization(run_dir)

    with np.load(data_path) as data:
        assert "jacobian_eigenvalues" in data.files
        assert "jacobian_spectral_abscissa" in data.files
    assert (run_dir / "figures/latent_jacobian/epoch-000001.png").is_file()
    assert (run_dir / "figures/latent_dynamics/epoch-000001.png").is_file()


def test_visualize_rerun_recomputes_with_analysis_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounds/grid overrides recompute latent dynamics from checkpoints."""
    monkeypatch.chdir(tmp_path)
    run_dir = run_rank2_training(SMOKE)

    rerun_rank2_visualization(
        run_dir,
        coordinate_bounds=(-3.0, 3.0, -2.0, 2.0),
        grid_points=9,
    )

    with np.load(run_dir / "diagnostics" / "latent_dynamics.npz") as data:
        np.testing.assert_allclose(
            data["bounds"], [-3.0, 3.0, -2.0, 2.0], rtol=1e-6
        )
        assert data["flow"].shape[1:] == (9, 9, 2)
    with (run_dir / "diagnostics" / "visualize_rerun.yaml").open(
        encoding="utf-8"
    ) as stream:
        marker = yaml.safe_load(stream)
    assert marker["kind"] == "visualize_rerun"
    assert marker["overrides"]["coordinate_bounds"] == [-3.0, 3.0, -2.0, 2.0]
    assert marker["overrides"]["grid_points"] == 9
