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


def test_short_training_run_writes_reproducible_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = run_rank2_training(SMOKE)
    assert run_dir == (tmp_path / "runs" / "phase1_smoke" / "seed-17").resolve()

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

    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    for expected_fragment in (
        "run started",
        "training started",
        "step=1",
        "training complete",
        "latent analysis complete",
        "vector field artifacts written",
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
        assert data["basis"].shape == (2, 8, 2)
        final_basis = torch.as_tensor(data["basis"][-1])
        torch.testing.assert_close(final_basis.T @ final_basis, torch.eye(2))

    experiment = resolve_experiment(SMOKE)
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


def test_run_tag_separates_run_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Learning-rate-style scans share a recipe but must not collide on disk."""
    monkeypatch.chdir(tmp_path)
    untagged = run_rank2_training(SMOKE)
    tagged = run_rank2_training(SMOKE, tag="lr1e-4")

    assert untagged == (tmp_path / "runs" / "phase1_smoke" / "seed-17").resolve()
    assert tagged == (
        tmp_path / "runs" / "phase1_smoke" / "lr1e-4-seed-17"
    ).resolve()
    assert tagged != untagged

    with (tagged / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["tag"] == "lr1e-4"


def test_recipe_tag_is_resolved(tmp_path: Path) -> None:
    recipe = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    recipe["tag"] = "lr1e-3"
    tagged_path = tmp_path / "tagged.yaml"
    tagged_path.write_text(
        yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8"
    )

    experiment = resolve_experiment(tagged_path)
    assert experiment.tag == "lr1e-3"
    assert experiment.default_run_dir(tmp_path / "runs") == (
        tmp_path / "runs" / "phase1_smoke" / "lr1e-3-seed-17"
    ).resolve()


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

    run_dir = tmp_path / "runs" / "phase1_smoke" / "seed-17"
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
