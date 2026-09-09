"""Delayed interval-categorization task generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ._shared import (
    TaskBatch,
    TaskConfigError,
    Trial,
    TrialMetadata,
    pair as _pair,
    to_steps as _to_steps,
)


@dataclass(frozen=True)
class SplitConfig:
    short_interval: tuple[float, float]
    long_interval: tuple[float, float]
    delay: tuple[float, float]
    s1_onset: tuple[float, float]


@dataclass(frozen=True)
class IntervalCategorizationConfig:
    name: str
    dt: float
    threshold: float
    stimulus_waveform: str
    stimulus_width: float
    stimulus_amplitude: float
    response_window: float
    pre_go_loss_weight: float
    response_loss_weight: float
    splits: dict[str, SplitConfig]

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, dt: float
    ) -> "IntervalCategorizationConfig":
        required = {
            "name",
            "threshold",
            "stimulus",
            "response_window",
            "pre_go_loss_weight",
            "response_loss_weight",
            "splits",
        }
        missing = required - set(data)
        if missing:
            raise TaskConfigError(f"Missing task fields: {', '.join(sorted(missing))}")
        if dt <= 0:
            raise TaskConfigError("dt must be positive")

        raw_splits = data["splits"]
        if not isinstance(raw_splits, Mapping) or not raw_splits:
            raise TaskConfigError("splits must be a non-empty mapping")
        raw_stimulus = data["stimulus"]
        if not isinstance(raw_stimulus, Mapping):
            raise TaskConfigError("stimulus must be a mapping")
        stimulus_fields = {"waveform", "width", "amplitude"}
        missing_stimulus = stimulus_fields - set(raw_stimulus)
        if missing_stimulus:
            raise TaskConfigError(
                "Missing stimulus fields: " + ", ".join(sorted(missing_stimulus))
            )

        splits: dict[str, SplitConfig] = {}
        for split_name, raw_split in raw_splits.items():
            if not isinstance(raw_split, Mapping):
                raise TaskConfigError(f"Split '{split_name}' must be a mapping")
            try:
                split = SplitConfig(
                    short_interval=_pair(raw_split["short_interval"]),
                    long_interval=_pair(raw_split["long_interval"]),
                    delay=_pair(raw_split["delay"]),
                    s1_onset=_pair(raw_split["s1_onset"]),
                )
            except KeyError as error:
                raise TaskConfigError(
                    f"Split '{split_name}' is missing field {error.args[0]!r}"
                ) from error
            splits[str(split_name)] = split

        config = cls(
            name=str(data["name"]),
            dt=float(dt),
            threshold=float(data["threshold"]),
            stimulus_waveform=str(raw_stimulus["waveform"]),
            stimulus_width=float(raw_stimulus["width"]),
            stimulus_amplitude=float(raw_stimulus["amplitude"]),
            response_window=float(data["response_window"]),
            pre_go_loss_weight=float(data["pre_go_loss_weight"]),
            response_loss_weight=float(data["response_loss_weight"]),
            splits=splits,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.name:
            raise TaskConfigError("name must not be empty")
        if self.stimulus_waveform != "square":
            raise TaskConfigError("Only the 'square' stimulus waveform is supported")
        if self.stimulus_amplitude <= 0:
            raise TaskConfigError("stimulus amplitude must be positive")
        if self.pre_go_loss_weight < 0 or self.response_loss_weight <= 0:
            raise TaskConfigError(
                "loss weights must be non-negative and response positive"
            )

        _to_steps(self.stimulus_width, self.dt, "stimulus.width")
        _to_steps(self.response_window, self.dt, "response_window")
        _to_steps(self.threshold, self.dt, "threshold")
        for split_name, split in self.splits.items():
            for field_name in ("short_interval", "long_interval", "delay", "s1_onset"):
                low, high = getattr(split, field_name)
                if low < 0 or high < low:
                    raise TaskConfigError(
                        f"Invalid {field_name} range in split '{split_name}'"
                    )
                _to_steps(low, self.dt, f"{split_name}.{field_name}[0]")
                _to_steps(high, self.dt, f"{split_name}.{field_name}[1]")
            if split.short_interval[1] >= self.threshold:
                raise TaskConfigError(
                    f"Short intervals must be below threshold in '{split_name}'"
                )
            if split.long_interval[0] <= self.threshold:
                raise TaskConfigError(
                    f"Long intervals must be above threshold in '{split_name}'"
                )
            minimum_separation = min(
                split.short_interval[0], split.long_interval[0], split.delay[0]
            )
            if self.stimulus_width > minimum_separation:
                raise TaskConfigError(
                    f"stimulus.width causes overlapping cues in split '{split_name}'"
                )


class IntervalCategorizationTask:
    """Generate balanced delayed interval-categorization trials."""

    input_names = ("S1", "S2", "Go")

    def __init__(self, config: IntervalCategorizationConfig, split: str = "train"):
        if split not in config.splits:
            available = ", ".join(sorted(config.splits))
            raise TaskConfigError(f"Unknown split '{split}'. Available: {available}")
        self.config = config
        self.split_name = split
        self.split = config.splits[split]

    def build_trial(self, *, interval: float, delay: float, s1_onset: float) -> Trial:
        """Build one trial from explicit relative timings."""
        cfg = self.config
        interval_steps = _to_steps(interval, cfg.dt, "interval")
        delay_steps = _to_steps(delay, cfg.dt, "delay")
        s1_step = _to_steps(s1_onset, cfg.dt, "s1_onset", allow_zero=True)
        threshold_steps = _to_steps(cfg.threshold, cfg.dt, "threshold")
        if interval_steps == threshold_steps:
            raise TaskConfigError("T == T_c is intentionally excluded")

        class_label = -1 if interval_steps < threshold_steps else 1
        s2_step = s1_step + interval_steps
        go_step = s2_step + delay_steps
        stimulus_steps = _to_steps(cfg.stimulus_width, cfg.dt, "stimulus.width")
        response_step = go_step + stimulus_steps
        response_steps = _to_steps(cfg.response_window, cfg.dt, "response_window")
        trial_steps = response_step + response_steps

        inputs = np.zeros((trial_steps, 3), dtype=np.float64)
        target = np.zeros((trial_steps, 1), dtype=np.float64)
        loss_mask = np.full(
            (trial_steps, 1), cfg.pre_go_loss_weight, dtype=np.float64
        )
        inputs[s1_step : s1_step + stimulus_steps, 0] = cfg.stimulus_amplitude
        inputs[s2_step : s2_step + stimulus_steps, 1] = cfg.stimulus_amplitude
        inputs[go_step : go_step + stimulus_steps, 2] = cfg.stimulus_amplitude
        target[response_step:, 0] = class_label
        loss_mask[response_step:, 0] = cfg.response_loss_weight

        metadata = TrialMetadata(
            s1_step=s1_step,
            s2_step=s2_step,
            go_step=go_step,
            response_step=response_step,
            trial_steps=trial_steps,
            interval=interval_steps * cfg.dt,
            delay=delay_steps * cfg.dt,
            class_label=class_label,
            split=self.split_name,
        )
        return Trial(inputs=inputs, target=target, loss_mask=loss_mask, metadata=metadata)

    def generate_batch(self, batch_size: int, rng: np.random.Generator) -> TaskBatch:
        """Sample a shuffled, class-balanced padded batch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size % 2:
            raise ValueError("batch_size must be even for exact class balance")

        n_short = batch_size // 2
        labels = np.concatenate(
            (
                np.full(n_short, -1, dtype=np.int8),
                np.full(batch_size - n_short, 1, dtype=np.int8),
            )
        )
        rng.shuffle(labels)

        trials: list[Trial] = []
        for label in labels:
            interval_range = (
                self.split.short_interval if label == -1 else self.split.long_interval
            )
            interval = self._sample_time(interval_range, rng)
            delay = self._sample_time(self.split.delay, rng)
            s1_onset = self._sample_time(self.split.s1_onset, rng)
            trials.append(
                self.build_trial(interval=interval, delay=delay, s1_onset=s1_onset)
            )

        return self.collate_trials(trials)

    def collate_trials(self, trials: Sequence[Trial]) -> TaskBatch:
        """Pad explicitly constructed trials into one batch."""
        if not trials:
            raise ValueError("trials must not be empty")
        batch_size = len(trials)
        max_steps = max(trial.metadata.trial_steps for trial in trials)
        inputs = np.zeros((batch_size, max_steps, 3), dtype=np.float64)
        target = np.zeros((batch_size, max_steps, 1), dtype=np.float64)
        loss_mask = np.zeros((batch_size, max_steps, 1), dtype=np.float64)
        valid_mask = np.zeros((batch_size, max_steps), dtype=np.bool_)
        for index, trial in enumerate(trials):
            length = trial.metadata.trial_steps
            inputs[index, :length] = trial.inputs
            target[index, :length] = trial.target
            loss_mask[index, :length] = trial.loss_mask
            valid_mask[index, :length] = True

        return TaskBatch(
            inputs=inputs,
            target=target,
            loss_mask=loss_mask,
            valid_mask=valid_mask,
            metadata=tuple(trial.metadata for trial in trials),
            dt=self.config.dt,
            input_names=self.input_names,
            loss_reduction="weighted_mean",
        )

    def evaluate_prediction(
        self, prediction: np.ndarray, batch: TaskBatch
    ) -> dict[str, float]:
        """Return the response-window categorization accuracy."""
        response = batch.target != 0
        predicted_class = np.where(prediction >= 0, 1.0, -1.0)
        return {
            "validation_accuracy": float(
                np.mean(predicted_class[response] == batch.target[response])
            )
        }

    def _sample_time(
        self, bounds: tuple[float, float], rng: np.random.Generator
    ) -> float:
        low = _to_steps(bounds[0], self.config.dt, "range lower bound", allow_zero=True)
        high = _to_steps(bounds[1], self.config.dt, "range upper bound", allow_zero=True)
        return int(rng.integers(low, high + 1)) * self.config.dt
