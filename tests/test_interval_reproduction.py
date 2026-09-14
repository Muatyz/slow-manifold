from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from slow_manifold.config import resolve_experiment
from slow_manifold.tasks import (
    IntervalReproductionConfig,
    IntervalReproductionTask,
    PhaseNormalizedLossConfig,
)
from slow_manifold.tasks.interval_reproduction import TaskConfigError

ROOT = Path(__file__).parents[1]


def make_task(split: str = "train") -> IntervalReproductionTask:
    experiment = resolve_experiment(
        ROOT / "experiments" / "phase1_rank2_IR.yaml"
    )
    config = IntervalReproductionConfig.from_mapping(
        experiment.components["task"], dt=experiment.components["model"]["dt"]
    )
    loss = PhaseNormalizedLossConfig.from_mapping(
        experiment.components["train"]["loss"]
    )
    return IntervalReproductionTask(config, split=split, loss=loss)


def test_paper_timing_and_shared_square_cues_are_encoded_exactly() -> None:
    task = make_task()
    trial = task.build_trial(interval=30.0, delay=20.0, s1_onset=50.0)
    metadata = trial.metadata

    assert metadata.s1_step == 50
    assert metadata.s2_step == 80
    assert metadata.go_step == 100
    assert metadata.response_step == 130
    assert metadata.trial_steps == 160
    assert metadata.class_label is None
    assert trial.inputs.shape == (160, 1)
    assert np.flatnonzero(trial.inputs[:, 0]).tolist() == [50, 80, 100]
    assert np.all(trial.target[:130] == 0.0)
    assert np.all(trial.target[130:] == 1.0)
    np.testing.assert_allclose(trial.loss_mask[:100].sum(), 1.0)
    np.testing.assert_allclose(trial.loss_mask[100:130].sum(), 1.0)
    np.testing.assert_allclose(trial.loss_mask[130:].sum(), 1.0)


def test_sampling_uses_paper_half_open_ranges_and_fixed_s1() -> None:
    task = make_task()
    batch = task.generate_batch(2048, np.random.default_rng(7))
    intervals = np.array([item.interval for item in batch.metadata])
    delays = np.array([item.delay for item in batch.metadata])

    assert intervals.min() >= 30.0
    assert intervals.max() <= 99.0
    assert delays.min() >= 20.0
    assert delays.max() <= 89.0
    assert {item.s1_step for item in batch.metadata} == {50}
    assert batch.input_names == ("cue",)
    assert batch.loss_reduction == "batch_mean"
    assert batch.inputs.shape[-1] == 1
    assert np.all(batch.loss_mask[~batch.valid_mask] == 0.0)


def test_reproduction_time_tracks_interval_not_delay() -> None:
    task = make_task()
    first = task.build_trial(interval=73.0, delay=20.0, s1_onset=50.0)
    second = task.build_trial(interval=73.0, delay=89.0, s1_onset=50.0)

    assert first.metadata.response_step - first.metadata.go_step == 73
    assert second.metadata.response_step - second.metadata.go_step == 73


@pytest.mark.parametrize("split", ["validation", "ood_short", "ood_long"])
def test_all_declared_splits_generate_valid_batches(split: str) -> None:
    batch = make_task(split).generate_batch(5, np.random.default_rng(3))
    assert len(batch.metadata) == 5
    assert np.all(batch.valid_mask.sum(axis=1) > 0)


def test_nonexclusive_sampling_semantics_are_rejected() -> None:
    experiment = resolve_experiment(
        ROOT / "experiments" / "phase0_reproduction_task_sanity.yaml"
    )
    task_mapping = deepcopy(experiment.components["task"])
    task_mapping["sampling"]["upper_bound"] = "inclusive"

    with pytest.raises(TaskConfigError, match="exclusive"):
        IntervalReproductionConfig.from_mapping(
            task_mapping, dt=experiment.components["model"]["dt"]
        )


def test_timing_metrics_use_first_post_go_sustained_crossing() -> None:
    task = make_task()
    batch = task.collate_trials(
        [
            task.build_trial(interval=30.0, delay=20.0, s1_onset=50.0),
            task.build_trial(interval=40.0, delay=20.0, s1_onset=50.0),
        ]
    )
    prediction = np.zeros_like(batch.target)
    prediction[0, 125:, 0] = 0.75  # five steps early
    metrics = task.evaluate_prediction(prediction, batch)

    assert metrics["validation_timing_mae"] == 5.0
    assert metrics["validation_timing_rmse"] == 5.0
    assert metrics["validation_timing_bias"] == -5.0
    assert metrics["validation_premature_rate"] == 0.5
    assert metrics["validation_no_response_rate"] == 0.5
