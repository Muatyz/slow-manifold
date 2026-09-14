from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from slow_manifold.config import resolve_experiment
from slow_manifold.tasks import (
    IntervalCategorizationConfig,
    IntervalCategorizationTask,
    PhaseNormalizedLossConfig,
)
from slow_manifold.tasks.interval_categorization import TaskConfigError

ROOT = Path(__file__).parents[1]


def make_task(split: str = "train") -> IntervalCategorizationTask:
    experiment = resolve_experiment(ROOT / "experiments" / "phase1_rank2_baseline.yaml")
    config = IntervalCategorizationConfig.from_mapping(
        experiment.components["task"], dt=experiment.components["model"]["dt"]
    )
    loss = PhaseNormalizedLossConfig.from_mapping(
        experiment.components["train"]["loss"]
    )
    return IntervalCategorizationTask(config, split=split, loss=loss)


def test_batch_is_balanced_and_padded_safely() -> None:
    task = make_task()
    batch = task.generate_batch(20, np.random.default_rng(12))

    labels = np.array([item.class_label for item in batch.metadata])
    assert batch.inputs.shape[:2] == batch.valid_mask.shape
    assert batch.inputs.shape[-1] == 1
    assert batch.input_names == ("cue",)
    assert batch.target.shape == batch.loss_mask.shape
    assert np.sum(labels == -1) == np.sum(labels == 1) == 10
    assert np.all(batch.loss_mask[~batch.valid_mask] == 0.0)


def test_odd_batch_size_is_rejected_to_preserve_balance() -> None:
    task = make_task()
    with pytest.raises(ValueError, match="even"):
        task.generate_batch(5, np.random.default_rng(12))


def test_trial_timing_target_and_mask_are_consistent() -> None:
    task = make_task()
    trial = task.build_trial(interval=3.2, delay=2.1, s1_onset=0.7)
    meta = trial.metadata
    stimulus_steps = round(task.config.stimulus.width / task.config.dt)

    assert meta.s2_step - meta.s1_step == 32
    assert meta.go_step - meta.s2_step == 21
    assert meta.response_step - meta.go_step == stimulus_steps
    assert meta.class_label == -1
    assert np.all(
        trial.inputs[meta.s1_step : meta.s1_step + stimulus_steps, 0] == 1.0
    )
    assert np.all(
        trial.inputs[meta.s2_step : meta.s2_step + stimulus_steps, 0] == 1.0
    )
    assert np.all(
        trial.inputs[meta.go_step : meta.go_step + stimulus_steps, 0] == 1.0
    )
    assert np.all(trial.target[: meta.response_step] == 0.0)
    assert np.all(trial.target[meta.response_step :] == -1.0)
    np.testing.assert_allclose(trial.loss_mask[: meta.response_step].sum(), 1.0)
    np.testing.assert_allclose(trial.loss_mask[meta.response_step :].sum(), 1.0)
    assert task.collate_trials([trial]).loss_reduction == "batch_mean"


def test_train_loss_lambdas_scale_normalized_task_phases() -> None:
    task = make_task()
    custom_loss = PhaseNormalizedLossConfig.from_mapping(
        {
            "name": "phase_normalized_mse",
            "phase_weights": [
                {"phase": "pre_response", "lambda": 2.5},
                {"phase": "response", "lambda": 0.75},
            ],
        }
    )
    weighted_task = IntervalCategorizationTask(task.config, loss=custom_loss)
    trial = weighted_task.build_trial(interval=3.2, delay=2.1, s1_onset=0.7)

    np.testing.assert_allclose(
        trial.loss_mask[: trial.metadata.response_step].sum(), 2.5
    )
    np.testing.assert_allclose(
        trial.loss_mask[trial.metadata.response_step :].sum(), 0.75
    )


def test_relative_interval_label_does_not_depend_on_onset_or_delay() -> None:
    task = make_task()
    first = task.build_trial(interval=6.4, delay=1.0, s1_onset=0.5)
    second = task.build_trial(interval=6.4, delay=3.8, s1_onset=1.4)

    assert first.metadata.class_label == second.metadata.class_label == 1
    assert first.metadata.s2_step - first.metadata.s1_step == 64
    assert second.metadata.s2_step - second.metadata.s1_step == 64


def test_threshold_interval_is_rejected() -> None:
    task = make_task()
    with pytest.raises(TaskConfigError, match="intentionally excluded"):
        task.build_trial(interval=5.0, delay=2.0, s1_onset=1.0)


def test_square_wave_width_is_configurable() -> None:
    experiment = resolve_experiment(ROOT / "experiments" / "phase0_task_sanity.yaml")
    task_mapping = deepcopy(experiment.components["task"])
    task_mapping["stimulus"]["width"] = 0.5
    config = IntervalCategorizationConfig.from_mapping(
        task_mapping, dt=experiment.components["model"]["dt"]
    )
    trial = IntervalCategorizationTask(config).build_trial(
        interval=3.0, delay=1.0, s1_onset=0.5
    )

    assert trial.inputs.shape[1] == 1
    assert np.count_nonzero(trial.inputs[:, 0]) == 15


def test_stimulus_encoding_must_be_explicit() -> None:
    experiment = resolve_experiment(ROOT / "experiments" / "phase0_task_sanity.yaml")
    task_mapping = deepcopy(experiment.components["task"])
    del task_mapping["stimulus"]["encoding"]

    with pytest.raises(TaskConfigError, match="encoding"):
        IntervalCategorizationConfig.from_mapping(
            task_mapping, dt=experiment.components["model"]["dt"]
        )


def test_separate_stimulus_channels_are_rejected() -> None:
    experiment = resolve_experiment(ROOT / "experiments" / "phase0_task_sanity.yaml")
    task_mapping = deepcopy(experiment.components["task"])
    task_mapping["stimulus"]["encoding"] = "separate"

    with pytest.raises(TaskConfigError, match="shared"):
        IntervalCategorizationConfig.from_mapping(
            task_mapping, dt=experiment.components["model"]["dt"]
        )


def test_unsupported_waveform_is_rejected() -> None:
    experiment = resolve_experiment(ROOT / "experiments" / "phase0_task_sanity.yaml")
    task_mapping = deepcopy(experiment.components["task"])
    task_mapping["stimulus"]["waveform"] = "sine"

    with pytest.raises(TaskConfigError, match="square"):
        IntervalCategorizationConfig.from_mapping(
            task_mapping, dt=experiment.components["model"]["dt"]
        )


def test_square_wave_width_cannot_overlap_cues() -> None:
    experiment = resolve_experiment(ROOT / "experiments" / "phase0_task_sanity.yaml")
    task_mapping = deepcopy(experiment.components["task"])
    task_mapping["stimulus"]["width"] = 1.1

    with pytest.raises(TaskConfigError, match="overlapping"):
        IntervalCategorizationConfig.from_mapping(
            task_mapping, dt=experiment.components["model"]["dt"]
        )


@pytest.mark.parametrize("split", ["validation", "ood_interval", "ood_delay", "ood_onset"])
def test_all_declared_splits_generate_valid_batches(split: str) -> None:
    task = make_task(split)
    batch = task.generate_batch(6, np.random.default_rng(3))
    assert len(batch.metadata) == 6
    assert {item.class_label for item in batch.metadata} == {-1, 1}
