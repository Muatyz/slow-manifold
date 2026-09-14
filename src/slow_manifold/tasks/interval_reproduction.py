"""Ramesan-style delayed interval-reproduction task generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ._shared import (
    PhaseNormalizedLossConfig,
    SquarePulseStimulusConfig,
    TaskBatch,
    TaskConfigError,
    Trial,
    TrialMetadata,
    pair,
    to_steps,
)


@dataclass(frozen=True)
class ReproductionSplitConfig:
    interval: tuple[float, float]
    delay: tuple[float, float]
    s1_onset: tuple[float, float]


@dataclass(frozen=True)
class IntervalReproductionConfig:
    """Configuration for cue-triggered reproduction of a sampled interval.

    Sampling ranges are defined on the ``dt`` grid with an exclusive upper
    bound, matching the paper's ``U[low, high)`` notation. Equal S1-onset
    bounds represent a fixed onset.
    """

    name: str
    dt: float
    stimulus: SquarePulseStimulusConfig
    target_baseline: float
    target_response: float
    response_threshold: float
    sustained_steps: int
    sampling_upper_bound: str
    splits: dict[str, ReproductionSplitConfig]

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, dt: float
    ) -> "IntervalReproductionConfig":
        try:
            stimulus = data["stimulus"]
            target = data["target"]
            evaluation = data["evaluation"]
            sampling = data["sampling"]
            raw_splits = data["splits"]
        except KeyError as error:
            raise TaskConfigError(f"Missing task field: {error.args[0]}") from error
        if not all(
            isinstance(item, Mapping)
            for item in (stimulus, target, evaluation, sampling, raw_splits)
        ):
            raise TaskConfigError(
                "stimulus, target, evaluation, sampling, and splits "
                "must be mappings"
            )
        if not raw_splits:
            raise TaskConfigError("splits must be a non-empty mapping")

        splits: dict[str, ReproductionSplitConfig] = {}
        for split_name, raw_split in raw_splits.items():
            if not isinstance(raw_split, Mapping):
                raise TaskConfigError(f"Split '{split_name}' must be a mapping")
            try:
                splits[str(split_name)] = ReproductionSplitConfig(
                    interval=pair(raw_split["interval"]),
                    delay=pair(raw_split["delay"]),
                    s1_onset=pair(raw_split["s1_onset"]),
                )
            except KeyError as error:
                raise TaskConfigError(
                    f"Split '{split_name}' is missing field {error.args[0]!r}"
                ) from error

        try:
            config = cls(
                name=str(data["name"]),
                dt=float(dt),
                stimulus=SquarePulseStimulusConfig.from_mapping(
                    stimulus, dt=dt
                ),
                target_baseline=float(target["baseline"]),
                target_response=float(target["response"]),
                response_threshold=float(evaluation["response_threshold"]),
                sustained_steps=int(evaluation["sustained_steps"]),
                sampling_upper_bound=str(sampling["upper_bound"]),
                splits=splits,
            )
        except KeyError as error:
            raise TaskConfigError(f"Missing task field: {error.args[0]}") from error
        config.validate()
        return config

    def validate(self) -> None:
        if not self.name:
            raise TaskConfigError("name must not be empty")
        if self.dt <= 0:
            raise TaskConfigError("dt must be positive")
        if self.sampling_upper_bound != "exclusive":
            raise TaskConfigError("Only exclusive sampling upper bounds are supported")
        if self.target_baseline == self.target_response:
            raise TaskConfigError("target baseline and response must differ")
        if not self.target_baseline < self.response_threshold < self.target_response:
            raise TaskConfigError("response_threshold must lie between target levels")
        if self.sustained_steps <= 0:
            raise TaskConfigError("sustained_steps must be positive")

        self.stimulus.validate(self.dt)
        for split_name, split in self.splits.items():
            for field_name in ("interval", "delay"):
                low, high = getattr(split, field_name)
                if low < 0 or high <= low:
                    raise TaskConfigError(
                        f"Invalid half-open {field_name} range in split '{split_name}'"
                    )
                to_steps(low, self.dt, f"{split_name}.{field_name}[0]")
                to_steps(high, self.dt, f"{split_name}.{field_name}[1]")
            onset_low, onset_high = split.s1_onset
            if onset_low < 0 or onset_high < onset_low:
                raise TaskConfigError(
                    f"Invalid s1_onset range in split '{split_name}'"
                )
            to_steps(
                onset_low,
                self.dt,
                f"{split_name}.s1_onset[0]",
                allow_zero=True,
            )
            to_steps(
                onset_high,
                self.dt,
                f"{split_name}.s1_onset[1]",
                allow_zero=True,
            )
            if self.stimulus.width > min(split.interval[0], split.delay[0]):
                raise TaskConfigError(
                    f"stimulus.width causes overlapping cues in split '{split_name}'"
                )


class IntervalReproductionTask:
    """Generate delayed interval-reproduction trials with one shared cue input."""

    def __init__(
        self,
        config: IntervalReproductionConfig,
        split: str = "train",
        loss: PhaseNormalizedLossConfig | None = None,
    ):
        if split not in config.splits:
            available = ", ".join(sorted(config.splits))
            raise TaskConfigError(f"Unknown split '{split}'. Available: {available}")
        self.config = config
        self.split_name = split
        self.split = config.splits[split]
        self.loss = loss or PhaseNormalizedLossConfig.unit_weights(
            "pre_go", "reproduction_wait", "response"
        )
        self.loss_weights = self.loss.require_phases(
            "pre_go", "reproduction_wait", "response"
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.config.stimulus.input_names

    def build_trial(self, *, interval: float, delay: float, s1_onset: float) -> Trial:
        """Build one trial; response onset is measured from Go onset."""
        cfg = self.config
        interval_steps = to_steps(interval, cfg.dt, "interval")
        delay_steps = to_steps(delay, cfg.dt, "delay")
        s1_step = to_steps(s1_onset, cfg.dt, "s1_onset", allow_zero=True)
        stimulus_steps = to_steps(cfg.stimulus.width, cfg.dt, "stimulus.width")
        if stimulus_steps > min(interval_steps, delay_steps):
            raise TaskConfigError("stimulus.width causes overlapping cues")

        s2_step = s1_step + interval_steps
        go_step = s2_step + delay_steps
        response_step = go_step + interval_steps
        # Dinc-style 2T production horizon: wait T after Go, then supervise a
        # response phase of the same duration.
        trial_steps = go_step + 2 * interval_steps

        inputs = cfg.stimulus.render(
            cue_steps=(s1_step, s2_step, go_step),
            trial_steps=trial_steps,
            dt=cfg.dt,
        )
        target = np.full(
            (trial_steps, 1), cfg.target_baseline, dtype=np.float64
        )
        loss_mask = np.zeros((trial_steps, 1), dtype=np.float64)
        target[response_step:, 0] = cfg.target_response
        # Each phase contributes its configured weight independent of duration.
        loss_mask[:go_step, 0] = self.loss_weights["pre_go"] / go_step
        loss_mask[go_step:response_step, 0] = (
            self.loss_weights["reproduction_wait"] / interval_steps
        )
        loss_mask[response_step:trial_steps, 0] = (
            self.loss_weights["response"] / interval_steps
        )

        metadata = TrialMetadata(
            s1_step=s1_step,
            s2_step=s2_step,
            go_step=go_step,
            response_step=response_step,
            trial_steps=trial_steps,
            interval=interval_steps * cfg.dt,
            delay=delay_steps * cfg.dt,
            split=self.split_name,
        )
        return Trial(
            inputs=inputs, target=target, loss_mask=loss_mask, metadata=metadata
        )

    def generate_batch(self, batch_size: int, rng: np.random.Generator) -> TaskBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        trials = [
            self.build_trial(
                interval=self._sample_half_open(self.split.interval, rng),
                delay=self._sample_half_open(self.split.delay, rng),
                s1_onset=self._sample_onset(self.split.s1_onset, rng),
            )
            for _ in range(batch_size)
        ]
        return self.collate_trials(trials)

    def collate_trials(self, trials: Sequence[Trial]) -> TaskBatch:
        if not trials:
            raise ValueError("trials must not be empty")
        batch_size = len(trials)
        max_steps = max(trial.metadata.trial_steps for trial in trials)
        inputs = np.zeros(
            (batch_size, max_steps, self.config.stimulus.input_size),
            dtype=np.float64,
        )
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
            loss_reduction="batch_mean",
        )

    def evaluate_prediction(
        self, prediction: np.ndarray, batch: TaskBatch
    ) -> dict[str, float]:
        """Measure the first sustained post-Go threshold crossing."""
        errors: list[float] = []
        premature = 0
        missing = 0
        for trial, metadata in enumerate(batch.metadata):
            trace = prediction[trial, : metadata.trial_steps, 0]
            crossing = self._first_sustained_crossing(trace, metadata.go_step)
            if crossing is None:
                missing += 1
                continue
            error = (crossing - metadata.response_step) * batch.dt
            errors.append(error)
            premature += int(crossing < metadata.response_step)
        batch_size = len(batch.metadata)
        errors_array = np.asarray(errors, dtype=np.float64)
        if errors_array.size:
            mae = float(np.mean(np.abs(errors_array)))
            rmse = float(np.sqrt(np.mean(errors_array**2)))
            bias = float(np.mean(errors_array))
        else:
            mae = rmse = bias = float("nan")
        return {
            "validation_timing_mae": mae,
            "validation_timing_rmse": rmse,
            "validation_timing_bias": bias,
            "validation_premature_rate": premature / batch_size,
            "validation_no_response_rate": missing / batch_size,
        }

    def _first_sustained_crossing(
        self, trace: np.ndarray, start_step: int
    ) -> int | None:
        above = trace >= self.config.response_threshold
        duration = self.config.sustained_steps
        for step in range(start_step, len(trace) - duration + 1):
            if np.all(above[step : step + duration]):
                return step
        return None

    def _sample_half_open(
        self, bounds: tuple[float, float], rng: np.random.Generator
    ) -> float:
        low = to_steps(bounds[0], self.config.dt, "range lower bound")
        high = to_steps(bounds[1], self.config.dt, "range upper bound")
        return int(rng.integers(low, high)) * self.config.dt

    def _sample_onset(
        self, bounds: tuple[float, float], rng: np.random.Generator
    ) -> float:
        low = to_steps(
            bounds[0], self.config.dt, "s1_onset lower bound", allow_zero=True
        )
        high = to_steps(
            bounds[1], self.config.dt, "s1_onset upper bound", allow_zero=True
        )
        if low == high:
            return low * self.config.dt
        return int(rng.integers(low, high)) * self.config.dt
