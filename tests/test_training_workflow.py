import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from slow_manifold.config import resolve_experiment
from slow_manifold.models import Rank2CTRNN, Rank2CTRNNConfig
from slow_manifold.training.checkpoint import load_checkpoint
from slow_manifold.visualization.training_dynamics import (
    _cue_driven_point_ranges,
    _cue_line_width,
    _cue_path_effects,
    _cue_trajectory_color,
    _validate_y_scale,
)
from slow_manifold.workflows import (
    rerun_rank2_visualization,
    resume_rank2_training,
    run_rank2_training,
)

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "tests" / "fixtures" / "phase1_smoke.yaml"
REPRODUCTION_SMOKE = ROOT / "experiments" / "phase1_rank2_IR_smoke.yaml"


def _write_modified_smoke_recipe(path: Path) -> None:
    recipe = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    recipe["components"] = {
        name: str((SMOKE.parent / reference).resolve())
        for name, reference in recipe["components"].items()
    }
    recipe["overrides"]["analysis"]["grid_points"] = 9
    recipe["overrides"]["visualization"]["arrow_width"] = 0.0008
    path.write_text(
        yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_cue_driven_ranges_include_the_input_generated_transition() -> None:
    inputs = np.zeros((9, 1), dtype=np.float64)
    inputs[2:4, 0] = 1.0
    inputs[6, 0] = 1.0

    assert _cue_driven_point_ranges(inputs) == [(1, 4), (5, 7)]


def test_task_b_cue_overlay_is_a_single_red_line() -> None:
    reproduction = {
        "task_name": np.asarray("delayed_interval_reproduction")
    }
    categorization = {
        "task_name": np.asarray("delayed_interval_categorization")
    }

    assert _cue_trajectory_color(reproduction) == "tab:red"
    assert _cue_path_effects(reproduction, 1.5) == ()
    assert _cue_line_width(reproduction, 1.5) < _cue_line_width(
        categorization, 1.5
    )
    assert _cue_trajectory_color(categorization) == "#F0E442"
    assert _cue_path_effects(categorization, 1.5)


def test_log_loss_axis_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive for a log y-axis"):
        _validate_y_scale(
            "log",
            (np.asarray([0.5, 0.0]), np.asarray([0.4, 0.3])),
            name="loss",
        )


def test_loss_axis_rejects_unknown_scale() -> None:
    with pytest.raises(ValueError, match="linear.*log"):
        _validate_y_scale(
            "symlog",
            (np.asarray([0.5]), np.asarray([0.4])),
            name="loss",
        )


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
        "diagnostics/config.yaml",
        "diagnostics/checkpoint_selection.yaml",
        "diagnostics/latent_dynamics.npz",
        "diagnostics/latent_dynamics.yaml",
        "figures/config.yaml",
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
        "train_seconds",
        "validation_seconds",
        "validation_ran",
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
    assert status["representative_epochs"] == [0, 1]
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
        "configuration Train: loss=phase_normalized_mse | lambda pre-response=1 | lambda response=1",
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
    for stream in (
        "rollout_initial_state_train",
        "rollout_neural_noise_train",
        "rollout_initial_state_validation",
        "rollout_neural_noise_validation",
    ):
        assert stream in payload["rng_state"]

    with np.load(run_dir / "diagnostics" / "latent_dynamics.npz") as data:
        assert data["epochs"].tolist() == [0, 1]
        assert data["trajectory"].shape[1] == 4
        np.testing.assert_allclose(
            data["trial_interval"], [3.0, 4.5, 5.5, 7.0]
        )
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
            assert data[field].shape == (4,)
        assert np.all(data["trial_s1_step"] < data["trial_s2_step"])
        assert np.all(data["trial_s2_step"] < data["trial_go_step"])
        assert np.all(data["trial_go_step"] < data["trial_response_step"])
        first_s2 = int(data["trial_s2_step"].min())
        shared_prefix = np.repeat(
            data["trajectory"][:, :1, :first_s2],
            repeats=4,
            axis=1,
        )
        np.testing.assert_allclose(
            data["trajectory"][:, :, :first_s2],
            shared_prefix,
            rtol=0.0,
            atol=0.0,
        )
        final_basis = torch.as_tensor(data["basis"][-1])
        torch.testing.assert_close(final_basis.T @ final_basis, torch.eye(2))
    with (run_dir / "diagnostics" / "latent_dynamics.yaml").open(
        encoding="utf-8"
    ) as stream:
        diagnostics_metadata = yaml.safe_load(stream)
    assert diagnostics_metadata["grid_points"] == 7
    with (run_dir / "diagnostics" / "checkpoint_selection.yaml").open(
        encoding="utf-8"
    ) as stream:
        checkpoint_selection = yaml.safe_load(stream)
    assert checkpoint_selection["selection_mode"] == "manual"
    assert [item["role"] for item in checkpoint_selection["selected"]] == [
        "initialization",
        "manual",
    ]
    with (run_dir / "diagnostics" / "config.yaml").open(
        encoding="utf-8"
    ) as stream:
        analysis_stage = yaml.safe_load(stream)
    with (run_dir / "figures" / "config.yaml").open(encoding="utf-8") as stream:
        visualization_stage = yaml.safe_load(stream)
    assert analysis_stage["kind"] == "analysis"
    assert analysis_stage["config_source"] == "training"
    assert analysis_stage["config"]["grid_points"] == 7
    assert visualization_stage["kind"] == "visualization"
    assert (
        visualization_stage["analysis_fingerprint"]
        == analysis_stage["fingerprint"]
    )
    assert visualization_stage["config"]["arrow_stride"] == 2

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
    outputs, _, _ = restored.rollout(
        torch.zeros(2, 5, model_config.input_size)
    )
    assert torch.isfinite(outputs).all()

    first_rng = np.random.default_rng()
    first_rng.bit_generator.state = payload["rng_state"]["task_train"]
    second_payload = load_checkpoint(run_dir / "checkpoints" / "final.pt")
    second_rng = np.random.default_rng()
    second_rng.bit_generator.state = second_payload["rng_state"]["task_train"]
    np.testing.assert_array_equal(first_rng.random(8), second_rng.random(8))


def test_validation_cadence_keeps_final_and_checkpoint_metrics(
    tmp_path: Path,
) -> None:
    recipe = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    recipe["components"] = {
        name: str((SMOKE.parent / reference).resolve())
        for name, reference in recipe["components"].items()
    }
    recipe["overrides"]["train"].update(
        {
            "epochs": 3,
            "validation_every": 2,
            "log_every": 1,
            "checkpoint": {"every": 3},
        }
    )
    recipe["overrides"]["analysis"]["representative_selection"] = {
        "mode": "manual",
        "manual_epochs": [0, 3],
    }
    recipe_path = tmp_path / "validation-cadence.yaml"
    recipe_path.write_text(
        yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    run_dir = run_rank2_training(recipe_path, tmp_path / "run")
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [int(row["validation_ran"]) for row in rows] == [1, 0, 1, 1]
    assert rows[1]["validation_loss"] == ""
    assert float(rows[1]["validation_seconds"]) == 0.0
    assert all(rows[index]["validation_loss"] for index in (0, 2, 3))
    assert (run_dir / "checkpoints" / "epoch-000003.pt").is_file()


def test_training_run_refuses_to_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        run_rank2_training(SMOKE, output_dir)


def test_interrupted_training_resumes_with_exact_model_and_rng_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    recipe["components"] = {
        name: str((SMOKE.parent / reference).resolve())
        for name, reference in recipe["components"].items()
    }
    recipe["overrides"]["train"].update(
        {
            "epochs": 3,
            "validation_every": 1,
            "checkpoint": {"every": 1},
        }
    )
    recipe["overrides"]["analysis"]["representative_selection"] = {
        "mode": "manual",
        "manual_epochs": [0, 3],
    }
    recipe["overrides"]["visualization"]["render_movies"] = False
    recipe_path = tmp_path / "resume-smoke.yaml"
    recipe_path.write_text(
        yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    import slow_manifold.training.trainer as trainer

    original_append = trainer._append_metric
    interrupted = False

    def _interrupt_after_durable_metric(
        path: Path,
        row: dict[str, float | int | None],
        *,
        create: bool,
    ) -> None:
        nonlocal interrupted
        original_append(path, row, create=create)
        if not interrupted and int(row["epoch"]) == 2:
            interrupted = True
            raise RuntimeError("synthetic interruption after metric write")

    monkeypatch.setattr(trainer, "_append_metric", _interrupt_after_durable_metric)
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_rank2_training(recipe_path, interrupted_dir)
    frozen_config = (interrupted_dir / "config.yaml").read_bytes()
    with (interrupted_dir / "status.yaml").open(encoding="utf-8") as stream:
        failed_status = yaml.safe_load(stream)
    assert failed_status["status"] == "failed"
    assert failed_status["last_step"] == 2

    monkeypatch.setattr(trainer, "_append_metric", original_append)
    assert resume_rank2_training(interrupted_dir) == interrupted_dir
    assert (interrupted_dir / "config.yaml").read_bytes() == frozen_config

    reference_dir = run_rank2_training(recipe_path, tmp_path / "reference")
    resumed_payload = load_checkpoint(interrupted_dir / "checkpoints" / "final.pt")
    reference_payload = load_checkpoint(reference_dir / "checkpoints" / "final.pt")
    assert resumed_payload["epoch"] == reference_payload["epoch"] == 3
    for name, expected in reference_payload["model_state"].items():
        torch.testing.assert_close(resumed_payload["model_state"][name], expected)
    for name in (
        "task_train",
        "torch_cpu",
        "rollout_initial_state_train",
        "rollout_neural_noise_train",
        "rollout_initial_state_validation",
        "rollout_neural_noise_validation",
    ):
        actual = resumed_payload["rng_state"][name]
        expected = reference_payload["rng_state"][name]
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected)
        else:
            assert actual == expected

    with (interrupted_dir / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        resumed_rows = list(csv.DictReader(stream))
    with (reference_dir / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reference_rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in resumed_rows] == [0, 1, 2, 3]
    timing_fields = {
        "train_seconds",
        "validation_seconds",
        "epoch_seconds",
        "elapsed_seconds",
    }
    for actual, expected in zip(resumed_rows, reference_rows):
        for field in set(actual) - timing_fields:
            if actual[field] == "":
                assert expected[field] == ""
            else:
                assert float(actual[field]) == pytest.approx(
                    float(expected[field]), rel=0.0, abs=0.0
                )

    with (interrupted_dir / "status.yaml").open(encoding="utf-8") as stream:
        completed_status = yaml.safe_load(stream)
    assert completed_status["status"] == "complete"
    assert completed_status["resume_count"] == 1
    assert completed_status["resumed_from_epoch"] == 1
    assert completed_status["previous_status"] == "failed"
    assert (
        "discarded_metric_rows=1"
        in (interrupted_dir / "run.log").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="already complete"):
        resume_rank2_training(interrupted_dir)


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
        assert data["trajectory"].shape[1] == 4
        np.testing.assert_allclose(
            data["trial_interval"], [30.0, 50.0, 75.0, 99.0]
        )
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
    original_config = (run_dir / "config.yaml").read_bytes()

    import slow_manifold.workflows.visualize_run as visualize_workflow

    def _must_not_recompute(*args: object, **kwargs: object) -> None:
        raise AssertionError("visualization-only rerun must reuse diagnostics")

    monkeypatch.setattr(visualize_workflow, "analyze_checkpoints", _must_not_recompute)
    figures_dir = rerun_rank2_visualization(
        run_dir,
        arrow_stride=1,
        arrow_length_fraction=0.0075,
        arrow_width=0.0012,
        trajectory_line_width=0.8,
    )

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
    # The complete latest figure config is recorded; the launch config is immutable.
    with (run_dir / "figures" / "config.yaml").open(encoding="utf-8") as stream:
        stage = yaml.safe_load(stream)
    assert stage["config_source"] == "current"
    assert stage["cli_overrides"] == {
        "arrow_stride": 1,
        "arrow_length_fraction": 0.0075,
        "arrow_width": 0.0012,
        "trajectory_line_width": 0.8,
    }
    assert stage["config"]["arrow_stride"] == 1
    assert stage["config"]["arrow_length_fraction"] == pytest.approx(0.0075)
    assert stage["config"]["arrow_width"] == pytest.approx(0.0012)
    assert stage["config"]["trajectory_line_width"] == pytest.approx(0.8)
    assert (run_dir / "config.yaml").read_bytes() == original_config
    frozen = yaml.safe_load(original_config)
    _, original_visualization, _, _ = visualize_workflow._resolve_stage_mappings(
        frozen,
        run_dir,
        config_source="original",
        experiment=None,
    )
    _, last_visualization, _, _ = visualize_workflow._resolve_stage_mappings(
        frozen,
        run_dir,
        config_source="last",
        experiment=None,
    )
    assert original_visualization["arrow_stride"] == 2
    assert last_visualization["arrow_stride"] == 1
    assert last_visualization["arrow_width"] == pytest.approx(0.0012)
    assert original_log.count("visualization rerun") == 0

    manual_figures = rerun_rank2_visualization(
        run_dir,
        tmp_path / "manual-figures",
        representative_epochs=[0],
    )
    with (run_dir / "diagnostics" / "checkpoint_selection.yaml").open(
        encoding="utf-8"
    ) as stream:
        manual_selection = yaml.safe_load(stream)
    assert manual_selection["source"] == "visualize:cli-manual"
    assert [item["epoch"] for item in manual_selection["selected"]] == [0]
    assert (manual_figures / "latent_dynamics/epoch-000000.png").is_file()
    assert not (manual_figures / "latent_dynamics/epoch-000001.png").exists()


def test_current_config_source_resolves_an_explicit_recipe(tmp_path: Path) -> None:
    """Current mode reads complete post-training components from a recipe."""
    from slow_manifold.workflows.visualize_run import _resolve_stage_mappings

    recipe_path = tmp_path / "current.yaml"
    _write_modified_smoke_recipe(recipe_path)
    frozen = resolve_experiment(SMOKE).as_dict()

    analysis, visualization, analysis_source, visualization_source = (
        _resolve_stage_mappings(
            frozen,
            tmp_path,
            config_source="current",
            experiment=recipe_path,
        )
    )

    assert analysis["grid_points"] == 9
    assert visualization["arrow_width"] == pytest.approx(0.0008)
    assert analysis_source == str(recipe_path.resolve())
    assert visualization_source == str(recipe_path.resolve())


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
        square_coordinate_bounds=True,
    )

    with np.load(run_dir / "diagnostics" / "latent_dynamics.npz") as data:
        np.testing.assert_allclose(
            data["bounds"], [-3.0, 3.0, -3.0, 3.0], rtol=1e-6
        )
        assert data["flow"].shape[1:] == (9, 9, 2)
        assert bool(data["square_coordinate_bounds"]) is True
        assert np.ptp(data["grid_x"]) == pytest.approx(np.ptp(data["grid_y"]))
    with (run_dir / "diagnostics" / "config.yaml").open(
        encoding="utf-8"
    ) as stream:
        stage = yaml.safe_load(stream)
    assert stage["kind"] == "analysis"
    assert stage["config_source"] == "current"
    assert stage["cli_overrides"]["coordinate_bounds"] == [
        -3.0,
        3.0,
        -2.0,
        2.0,
    ]
    assert stage["config"]["grid_points"] == 9
    assert stage["config"]["square_coordinate_bounds"] is True
