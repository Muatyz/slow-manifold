"""Training loop for configured low-rank CTRNN experiments."""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch
from torch import Tensor

from slow_manifold.models import Rank2CTRNN
from slow_manifold.tasks import ConfiguredTask, PhaseNormalizedLossConfig, TaskBatch
from slow_manifold.training.checkpoint import load_checkpoint, save_checkpoint
from slow_manifold.utils.logging import get_logger


class TrainConfigError(ValueError):
    """Raised when training configuration is invalid."""


@dataclass(frozen=True)
class InitialStateConfig:
    """Distribution of the firing-rate state at the start of each trial."""

    distribution: str = "tanh_normal"
    mean: float = 0.0
    std: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "InitialStateConfig":
        values = {} if data is None else data
        if not isinstance(values, Mapping):
            raise TrainConfigError("rollout.initial_state must be a mapping")
        config = cls(
            distribution=str(values.get("distribution", "tanh_normal")),
            mean=float(values.get("mean", 0.0)),
            std=float(values.get("std", 0.0)),
        )
        if config.distribution != "tanh_normal":
            raise TrainConfigError("Only tanh_normal initial states are supported")
        if not isfinite(config.mean) or not isfinite(config.std) or config.std < 0:
            raise TrainConfigError("Initial-state mean/std must be finite and std >= 0")
        return config


@dataclass(frozen=True)
class NeuralNoiseConfig:
    """Gaussian noise added to the activation input at every Euler step."""

    distribution: str = "normal"
    mean: float = 0.0
    std: float = 0.0
    injection: str = "activation_input"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "NeuralNoiseConfig":
        values = {} if data is None else data
        if not isinstance(values, Mapping):
            raise TrainConfigError("rollout.neural_noise must be a mapping")
        config = cls(
            distribution=str(values.get("distribution", "normal")),
            mean=float(values.get("mean", 0.0)),
            std=float(values.get("std", 0.0)),
            injection=str(values.get("injection", "activation_input")),
        )
        if config.distribution != "normal":
            raise TrainConfigError("Only normal neural noise is supported")
        if config.injection != "activation_input":
            raise TrainConfigError("Only activation_input neural noise is supported")
        if not isfinite(config.mean) or not isfinite(config.std) or config.std < 0:
            raise TrainConfigError("Neural-noise mean/std must be finite and std >= 0")
        return config


@dataclass(frozen=True)
class RolloutContextConfig:
    sample_initial_state: bool
    apply_neural_noise: bool

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
        *,
        default_initial_state: bool,
        default_neural_noise: bool,
        name: str,
    ) -> "RolloutContextConfig":
        values = {} if data is None else data
        if not isinstance(values, Mapping):
            raise TrainConfigError(f"rollout.contexts.{name} must be a mapping")
        initial_state = values.get("sample_initial_state", default_initial_state)
        neural_noise = values.get("apply_neural_noise", default_neural_noise)
        if not isinstance(initial_state, bool) or not isinstance(neural_noise, bool):
            raise TrainConfigError(f"rollout.contexts.{name} flags must be boolean")
        return cls(
            sample_initial_state=initial_state,
            apply_neural_noise=neural_noise,
        )


@dataclass(frozen=True)
class RolloutConfig:
    initial_state: InitialStateConfig
    neural_noise: NeuralNoiseConfig
    train: RolloutContextConfig
    validation: RolloutContextConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RolloutConfig":
        values = {} if data is None else data
        if not isinstance(values, Mapping):
            raise TrainConfigError("rollout must be a mapping")
        contexts = values.get("contexts", {})
        if not isinstance(contexts, Mapping):
            raise TrainConfigError("rollout.contexts must be a mapping")
        return cls(
            initial_state=InitialStateConfig.from_mapping(values.get("initial_state")),
            neural_noise=NeuralNoiseConfig.from_mapping(values.get("neural_noise")),
            train=RolloutContextConfig.from_mapping(
                contexts.get("train"),
                default_initial_state=True,
                default_neural_noise=True,
                name="train",
            ),
            validation=RolloutContextConfig.from_mapping(
                contexts.get("validation"),
                default_initial_state=True,
                default_neural_noise=False,
                name="validation",
            ),
        )


@dataclass(frozen=True)
class TrainConfig:
    name: str
    device: str
    epochs: int
    batch_size: int
    validation_batch_size: int
    validation_every: int
    log_every: int
    optimizer_name: str
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    gradient_clip_norm: float | None
    checkpoint_every: int
    loss: PhaseNormalizedLossConfig
    rollout: RolloutConfig
    num_threads: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainConfig":
        optimizer = data.get("optimizer")
        checkpoint = data.get("checkpoint")
        if not isinstance(optimizer, Mapping) or not isinstance(checkpoint, Mapping):
            raise TrainConfigError("optimizer and checkpoint must be mappings")
        try:
            clip = data.get("gradient_clip_norm")
            num_threads = data.get("num_threads")
            config = cls(
                name=str(data["name"]),
                device=str(data["device"]),
                epochs=int(data["epochs"]),
                batch_size=int(data["batch_size"]),
                validation_batch_size=int(data["validation_batch_size"]),
                validation_every=int(data.get("validation_every", 1)),
                log_every=int(data["log_every"]),
                optimizer_name=str(optimizer["name"]),
                learning_rate=float(optimizer["learning_rate"]),
                betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
                weight_decay=float(optimizer["weight_decay"]),
                gradient_clip_norm=None if clip is None else float(clip),
                checkpoint_every=int(checkpoint["every"]),
                loss=PhaseNormalizedLossConfig.from_mapping(data.get("loss")),
                rollout=RolloutConfig.from_mapping(data.get("rollout")),
                num_threads=None if num_threads is None else int(num_threads),
            )
        except (KeyError, IndexError, TypeError) as error:
            raise TrainConfigError(f"Invalid training configuration: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise TrainConfigError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise TrainConfigError("CUDA was requested but is not available")
        if (
            self.epochs <= 0
            or self.validation_every <= 0
            or self.log_every <= 0
            or self.checkpoint_every <= 0
        ):
            raise TrainConfigError("epochs and frequencies must be positive")
        if self.batch_size <= 0:
            raise TrainConfigError("batch_size must be positive")
        if self.validation_batch_size <= 0:
            raise TrainConfigError("validation_batch_size must be positive")
        if self.optimizer_name != "adam":
            raise TrainConfigError("Only the Adam optimizer is supported")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise TrainConfigError("Invalid optimizer scale")
        if not all(0 <= beta < 1 for beta in self.betas):
            raise TrainConfigError("Adam betas must lie in [0, 1)")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise TrainConfigError("gradient_clip_norm must be positive or null")
        if self.num_threads is not None and self.num_threads < 1:
            raise TrainConfigError("num_threads must be positive or null")


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[dict[str, float | int | None], ...]
    checkpoint_paths: dict[int, Path]
    best_epoch: int


def train_model(
    *,
    model: Rank2CTRNN,
    train_task: ConfiguredTask,
    validation_batch: TaskBatch,
    train_rng: np.random.Generator,
    rollout_seeds: Mapping[str, int],
    config: TrainConfig,
    run_dir: Path,
    snapshot_epochs: Sequence[int],
    resolved_config: Mapping[str, Any],
    resume_payload: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train or resume while recording Dinc-style optimization diagnostics."""
    if config.num_threads is not None:
        # Cap the PyTorch intra-op thread pool per the resolved train config.
        # Wall-clock is insensitive to this value on CUDA (benchmarked), while a
        # lower cap frees CPU cores for concurrent multi-seed runs.  This only
        # touches the current process; interop threads and global defaults are
        # left untouched so the rest of the session is unaffected.
        torch.set_num_threads(config.num_threads)
    device = torch.device(config.device)
    model.to(device)
    required_rollout_streams = {
        "initial_state_train",
        "neural_noise_train",
        "initial_state_validation",
        "neural_noise_validation",
    }
    missing_streams = required_rollout_streams - set(rollout_seeds)
    if missing_streams:
        raise ValueError(
            "Missing rollout RNG seeds: " + ", ".join(sorted(missing_streams))
        )
    rollout_generators = {
        name: torch.Generator(device=device).manual_seed(int(rollout_seeds[name]))
        for name in sorted(required_rollout_streams)
    }
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    requested_snapshots = {int(epoch) for epoch in snapshot_epochs}
    requested_snapshots.update({0, config.epochs})
    checkpoint_paths: dict[int, Path] = {}
    history: list[dict[str, float | int | None]] = []
    metrics_path = run_dir / "metrics.csv"
    logger = get_logger("train")
    logger.info(
        "training %s epochs=%d batch_size=%d validation_every=%d "
        "device=%s learning_rate=%s initial_state_std=%g neural_noise_std=%g "
        "torch_intra_threads=%d",
        "resuming" if resume_payload is not None else "started",
        config.epochs,
        config.batch_size,
        config.validation_every,
        config.device,
        config.learning_rate,
        config.rollout.initial_state.std,
        config.rollout.neural_noise.std,
        torch.get_num_threads(),
    )
    training_started = time.perf_counter()
    elapsed_seconds = 0.0
    step_seconds: deque[float] = deque(maxlen=100)
    validation_initial_state = _sample_context_initial_state(
        model,
        batch_size=validation_batch.inputs.shape[0],
        initial_state=config.rollout.initial_state,
        enabled=config.rollout.validation.sample_initial_state,
        generator=rollout_generators["initial_state_validation"],
    )

    def _measured_step(
        *, batch: TaskBatch, epoch: int, update: bool, validate: bool
    ) -> dict[str, float | int | None]:
        """Run one measured step and append wall-clock timing to its row."""
        nonlocal elapsed_seconds
        start = time.perf_counter()
        row = _measure_step(
            model=model,
            batch=batch,
            validation_batch=validation_batch,
            optimizer=optimizer,
            update=update,
            clip_norm=config.gradient_clip_norm,
            epoch=epoch,
            device=device,
            task=train_task,
            rollout=config.rollout,
            validation_initial_state=validation_initial_state,
            rollout_generators=rollout_generators,
            validate=validate,
        )
        duration = time.perf_counter() - start
        elapsed_seconds += duration
        row["epoch_seconds"] = duration
        row["elapsed_seconds"] = elapsed_seconds
        return row

    start_epoch = 0
    metric_fields: tuple[str, ...]
    best_epoch: int
    best_loss: float
    if resume_payload is not None:
        start_epoch = _restore_training_state(
            payload=resume_payload,
            model=model,
            optimizer=optimizer,
            train_rng=train_rng,
            rollout_generators=rollout_generators,
            target_epoch=config.epochs,
        )
        best_payload = load_checkpoint(run_dir / "checkpoints" / "best.pt")
        best_epoch = int(best_payload["epoch"])
        if best_epoch > start_epoch:
            raise ValueError(
                "best.pt is newer than the selected resume checkpoint; "
                "resume selection is inconsistent"
            )
        best_metric = best_payload.get("metrics", {}).get("validation_loss")
        if best_metric is None:
            raise ValueError("best.pt does not contain a validation_loss")
        best_loss = float(best_metric)
        history, discarded_rows = _read_metrics_for_resume(
            metrics_path, checkpoint_epoch=start_epoch
        )
        metric_fields = tuple(history[0])
        elapsed_seconds = float(history[-1].get("elapsed_seconds") or 0.0)
        step_seconds.extend(
            float(row["epoch_seconds"])
            for row in history[-100:]
            if row.get("epoch_seconds") is not None
        )
        checkpoint_paths = _existing_epoch_checkpoint_paths(
            run_dir, maximum_epoch=start_epoch
        )
        logger.info(
            "training state restored checkpoint_epoch=%d target_epoch=%d "
            "metrics_rows=%d discarded_metric_rows=%d best_epoch=%d",
            start_epoch,
            config.epochs,
            len(history),
            discarded_rows,
            best_epoch,
        )
    else:
        initial_batch = train_task.generate_batch(config.batch_size, train_rng)
        initial_row = _measured_step(
            batch=initial_batch, epoch=0, update=False, validate=True
        )
        metric_fields = tuple(initial_row)
        history.append(initial_row)
        _append_metric(metrics_path, initial_row, create=True)
        checkpoint_paths[0] = _save_epoch_checkpoint(
            model,
            optimizer,
            train_rng,
            rollout_generators,
            config,
            history,
            resolved_config,
            run_dir,
            0,
        )
        save_checkpoint(
            _checkpoint_payload(
                model,
                optimizer,
                train_rng,
                rollout_generators,
                config,
                history,
                resolved_config,
                0,
            ),
            run_dir / "checkpoints" / "initial.pt",
        )

        best_epoch = 0
        best_loss = float(initial_row["validation_loss"])
        save_checkpoint(
            _checkpoint_payload(
                model,
                optimizer,
                train_rng,
                rollout_generators,
                config,
                history,
                resolved_config,
                0,
            ),
            run_dir / "checkpoints" / "best.pt",
        )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        should_snapshot = (
            epoch % config.checkpoint_every == 0
            or epoch in requested_snapshots
            or epoch == config.epochs
        )
        should_validate = (
            epoch % config.validation_every == 0
            or should_snapshot
            or epoch == config.epochs
        )
        batch = train_task.generate_batch(config.batch_size, train_rng)
        measured = _measured_step(
            batch=batch,
            epoch=epoch,
            update=True,
            validate=should_validate,
        )
        row = {field: measured.get(field) for field in metric_fields}
        step_seconds.append(row["epoch_seconds"])
        history.append(row)
        _append_metric(metrics_path, row, create=False)

        validation_loss = row["validation_loss"]
        if validation_loss is not None and float(validation_loss) < best_loss:
            best_loss = float(validation_loss)
            best_epoch = epoch
            save_checkpoint(
                _checkpoint_payload(
                    model,
                    optimizer,
                    train_rng,
                    rollout_generators,
                    config,
                    history,
                    resolved_config,
                    epoch,
                ),
                run_dir / "checkpoints" / "best.pt",
            )

        if should_snapshot:
            start = time.perf_counter()
            checkpoint_paths[epoch] = _save_epoch_checkpoint(
                model,
                optimizer,
                train_rng,
                rollout_generators,
                config,
                history,
                resolved_config,
                run_dir,
                epoch,
            )
            logger.info(
                "checkpoint saved step=%d file=epoch-%06d.pt seconds=%.2fs",
                epoch,
                epoch,
                time.perf_counter() - start,
            )
        if epoch % config.log_every == 0 or epoch == config.epochs:
            _log_training_progress(
                logger=logger,
                row=row,
                epoch=epoch,
                eta_seconds=_eta_seconds(config.epochs, epoch, step_seconds),
            )

    final_payload = _checkpoint_payload(
        model,
        optimizer,
        train_rng,
        rollout_generators,
        config,
        history,
        resolved_config,
        config.epochs,
    )
    start = time.perf_counter()
    save_checkpoint(final_payload, run_dir / "checkpoints" / "final.pt")
    logger.info(
        "checkpoint saved step=%d file=final.pt seconds=%.2fs",
        config.epochs,
        time.perf_counter() - start,
    )
    logger.info(
        "training complete epochs=%d wall_seconds=%.1fs "
        "median_epoch_seconds=%.3fs best_epoch=%d",
        config.epochs,
        time.perf_counter() - training_started,
        _median_seconds(step_seconds),
        best_epoch,
    )
    return TrainingResult(
        history=tuple(history),
        checkpoint_paths=checkpoint_paths,
        best_epoch=best_epoch,
    )


def _measure_step(
    *,
    model: Rank2CTRNN,
    batch: TaskBatch,
    validation_batch: TaskBatch,
    optimizer: torch.optim.Optimizer,
    update: bool,
    clip_norm: float | None,
    epoch: int,
    device: torch.device,
    task: ConfiguredTask,
    rollout: RolloutConfig,
    validation_initial_state: Tensor | None,
    rollout_generators: Mapping[str, torch.Generator],
    validate: bool,
) -> dict[str, float | int | None]:
    train_started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train = _torch_batch(batch, device, model.config.torch_dtype)
    train_initial_state = _sample_context_initial_state(
        model,
        batch_size=batch.inputs.shape[0],
        initial_state=rollout.initial_state,
        enabled=rollout.train.sample_initial_state,
        generator=rollout_generators["initial_state_train"],
    )
    train_noise = rollout.neural_noise
    prediction, _, _ = model.rollout(
        train["inputs"],
        initial_state=train_initial_state,
        neural_noise_mean=(
            train_noise.mean if rollout.train.apply_neural_noise else 0.0
        ),
        neural_noise_std=(
            train_noise.std if rollout.train.apply_neural_noise else 0.0
        ),
        generator=rollout_generators["neural_noise_train"],
    )
    train_loss = _masked_mse(
        prediction,
        train["target"],
        train["loss_mask"],
        reduction=batch.loss_reduction,
        batch_size=batch.inputs.shape[0],
    )
    train_loss.backward()
    recurrent_gradient_norm = _recurrent_gradient_norm(model)
    total_gradient_norm = _total_gradient_norm(model)
    if update:
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

    with torch.no_grad():
        parameter_diagnostics = _parameter_diagnostics(model)
    result: dict[str, float | int | None] = {
        "epoch": epoch,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "train_loss": float(train_loss.detach().cpu()),
        "recurrent_gradient_norm": recurrent_gradient_norm,
        "total_gradient_norm": total_gradient_norm,
        **parameter_diagnostics,
        "train_seconds": time.perf_counter() - train_started,
        "validation_ran": int(validate),
    }
    if not validate:
        result["validation_loss"] = None
        result["validation_seconds"] = 0.0
        return result

    validation_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        validation = _torch_batch(
            validation_batch, device, model.config.torch_dtype
        )
        validation_noise = rollout.neural_noise
        validation_prediction, _, _ = model.rollout(
            validation["inputs"],
            initial_state=validation_initial_state,
            neural_noise_mean=(
                validation_noise.mean
                if rollout.validation.apply_neural_noise
                else 0.0
            ),
            neural_noise_std=(
                validation_noise.std
                if rollout.validation.apply_neural_noise
                else 0.0
            ),
            generator=rollout_generators["neural_noise_validation"],
        )
        validation_loss = _masked_mse(
            validation_prediction,
            validation["target"],
            validation["loss_mask"],
            reduction=validation_batch.loss_reduction,
            batch_size=validation_batch.inputs.shape[0],
        )
        behavior_metrics = task.evaluate_prediction(
            validation_prediction.detach().cpu().numpy(), validation_batch
        )
    result.update(
        {
            "validation_loss": float(validation_loss.cpu()),
            **behavior_metrics,
            "validation_seconds": time.perf_counter() - validation_started,
        }
    )
    return result


def _sample_context_initial_state(
    model: Rank2CTRNN,
    *,
    batch_size: int,
    initial_state: InitialStateConfig,
    enabled: bool,
    generator: torch.Generator,
) -> Tensor | None:
    if not enabled:
        return None
    return model.sample_initial_state(
        batch_size,
        mean=initial_state.mean,
        std=initial_state.std,
        generator=generator,
    )


def _torch_batch(
    batch: TaskBatch, device: torch.device, dtype: torch.dtype
) -> dict[str, Tensor]:
    return {
        "inputs": torch.as_tensor(batch.inputs, dtype=dtype, device=device),
        "target": torch.as_tensor(batch.target, dtype=dtype, device=device),
        "loss_mask": torch.as_tensor(batch.loss_mask, dtype=dtype, device=device),
    }


def _masked_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    reduction: str = "weighted_mean",
    batch_size: int | None = None,
) -> Tensor:
    weighted_error = torch.sum(mask * (prediction - target).square())
    if reduction == "weighted_mean":
        return weighted_error / torch.sum(mask)
    if reduction == "batch_mean" and batch_size is not None and batch_size > 0:
        return weighted_error / batch_size
    raise ValueError(f"Unsupported loss reduction: {reduction!r}")


def _response_accuracy(prediction: Tensor, target: Tensor) -> Tensor:
    response = target != 0
    predicted_class = torch.where(prediction >= 0, 1.0, -1.0)
    return (predicted_class[response] == target[response]).to(torch.float32).mean()


def _recurrent_gradient_norm(model: Rank2CTRNN) -> float:
    norms = [
        torch.linalg.vector_norm(parameter.grad).detach()
        for parameter in (model.m, model.n)
        if parameter.grad is not None
    ]
    return float(torch.stack(norms).sum().cpu())


def _total_gradient_norm(model: Rank2CTRNN) -> float:
    squared = [
        parameter.grad.detach().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return float(torch.sqrt(torch.stack(squared).sum()).cpu())


def _parameter_diagnostics(model: Rank2CTRNN) -> dict[str, float]:
    # M=Qm Rm and N=Qn Rn, so the non-zero singular values of M N^T
    # equal those of the small rank-by-rank matrix Rm Rn^T.
    r_m = torch.linalg.qr(model.m, mode="reduced").R
    r_n = torch.linalg.qr(model.n, mode="reduced").R
    singular_values = torch.linalg.svdvals(r_m @ r_n.T)
    diagnostics = {
        "m_norm": float(torch.linalg.vector_norm(model.m).cpu()),
        "n_norm": float(torch.linalg.vector_norm(model.n).cpu()),
        "w_norm": float(torch.linalg.vector_norm(singular_values).cpu()),
    }
    diagnostics.update(
        {
            f"singular_value_{index + 1}": float(value.cpu())
            for index, value in enumerate(singular_values)
        }
    )
    return diagnostics


def _checkpoint_payload(
    model: Rank2CTRNN,
    optimizer: torch.optim.Optimizer,
    train_rng: np.random.Generator,
    rollout_generators: Mapping[str, torch.Generator],
    config: TrainConfig,
    history: Sequence[dict[str, float | int | None]],
    resolved_config: Mapping[str, Any],
    epoch: int,
) -> dict:
    return {
        "format_version": 2,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_config": asdict(config),
        "resolved_config": dict(resolved_config),
        "metrics": dict(history[-1]),
        "rng_state": {
            "task_train": train_rng.bit_generator.state,
            "torch_cpu": torch.random.get_rng_state(),
            **{
                f"rollout_{name}": generator.get_state()
                for name, generator in rollout_generators.items()
            },
        },
    }


def _save_epoch_checkpoint(
    model: Rank2CTRNN,
    optimizer: torch.optim.Optimizer,
    train_rng: np.random.Generator,
    rollout_generators: Mapping[str, torch.Generator],
    config: TrainConfig,
    history: Sequence[dict[str, float | int | None]],
    resolved_config: Mapping[str, Any],
    run_dir: Path,
    epoch: int,
) -> Path:
    return save_checkpoint(
        _checkpoint_payload(
            model,
            optimizer,
            train_rng,
            rollout_generators,
            config,
            history,
            resolved_config,
            epoch,
        ),
        run_dir / "checkpoints" / f"epoch-{epoch:06d}.pt",
    )


def _restore_training_state(
    *,
    payload: Mapping[str, Any],
    model: Rank2CTRNN,
    optimizer: torch.optim.Optimizer,
    train_rng: np.random.Generator,
    rollout_generators: Mapping[str, torch.Generator],
    target_epoch: int,
) -> int:
    """Restore every state that can influence subsequent optimization."""
    required = {"epoch", "model_state", "optimizer_state", "rng_state"}
    missing = required - set(payload)
    if missing:
        raise ValueError(
            "Resume checkpoint is missing: " + ", ".join(sorted(missing))
        )
    epoch = int(payload["epoch"])
    if epoch < 0 or epoch > target_epoch:
        raise ValueError(
            f"Resume checkpoint epoch {epoch} is outside target range 0..{target_epoch}"
        )
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping):
        raise ValueError("Resume checkpoint rng_state must be a mapping")
    required_rng = {"task_train", "torch_cpu"} | {
        f"rollout_{name}" for name in rollout_generators
    }
    missing_rng = required_rng - set(rng_state)
    if missing_rng:
        raise ValueError(
            "Resume checkpoint is missing RNG streams: "
            + ", ".join(sorted(missing_rng))
        )

    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    train_rng.bit_generator.state = rng_state["task_train"]
    torch.random.set_rng_state(rng_state["torch_cpu"])
    for name, generator in rollout_generators.items():
        generator.set_state(rng_state[f"rollout_{name}"])
    return epoch


def _read_metrics_for_resume(
    path: Path, *, checkpoint_epoch: int
) -> tuple[list[dict[str, float | int | None]], int]:
    """Validate metrics and discard rows newer than the restored checkpoint."""
    if not path.is_file():
        raise ValueError(f"Cannot resume without metrics.csv: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("metrics.csv has no header")
        fieldnames = list(reader.fieldnames)
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("metrics.csv has no data rows")
    try:
        epochs = [int(row["epoch"]) for row in raw_rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("metrics.csv has invalid epoch values") from error
    if epochs != sorted(set(epochs)):
        raise ValueError("metrics.csv epochs must be strictly increasing")

    kept = [row for row, epoch in zip(raw_rows, epochs) if epoch <= checkpoint_epoch]
    if not kept or int(kept[-1]["epoch"]) != checkpoint_epoch:
        raise ValueError(
            "metrics.csv does not contain the selected resume checkpoint epoch "
            f"{checkpoint_epoch}"
        )
    discarded = len(raw_rows) - len(kept)
    if discarded:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return [_coerce_metric_row(row) for row in kept], discarded


def _coerce_metric_row(
    row: Mapping[str, str]
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for key, value in row.items():
        if value == "":
            result[key] = None
        elif key in {"epoch", "validation_ran"}:
            result[key] = int(value)
        else:
            result[key] = float(value)
    return result


def _existing_epoch_checkpoint_paths(
    run_dir: Path, *, maximum_epoch: int
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    checkpoint_dir = run_dir / "checkpoints"
    for path in checkpoint_dir.glob("epoch-*.pt"):
        try:
            epoch = int(path.stem.removeprefix("epoch-"))
        except ValueError:
            continue
        if epoch <= maximum_epoch:
            paths[epoch] = path
    return dict(sorted(paths.items()))


def _append_metric(
    path: Path, row: Mapping[str, float | int | None], *, create: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if create else "a"
    with path.open(mode, encoding="utf-8", newline="") as stream:
        fields = list(row)
        writer = csv.DictWriter(stream, fieldnames=fields)
        if create:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def _log_training_progress(
    *,
    logger: Any,
    row: Mapping[str, float | int | None],
    epoch: int,
    eta_seconds: float,
) -> None:
    validation_loss = row.get("validation_loss")
    if validation_loss is None:
        validation_summary = "validation=skipped"
    else:
        behavior = " ".join(
            f"{key.removeprefix('validation_')}={float(value):.3g}"
            for key, value in row.items()
            if key.startswith("validation_")
            and key
            not in {"validation_loss", "validation_ran", "validation_seconds"}
            and value is not None
        )
        validation_summary = f"validation_loss={float(validation_loss):.6g}"
        if behavior:
            validation_summary += f" {behavior}"
    logger.info(
        "step=%d train_loss=%.6g %s train_seconds=%.3fs "
        "validation_seconds=%.3fs epoch_seconds=%.3fs elapsed=%.1fs eta=%s",
        epoch,
        float(row["train_loss"]),
        validation_summary,
        float(row["train_seconds"]),
        float(row["validation_seconds"]),
        float(row["epoch_seconds"]),
        float(row["elapsed_seconds"]),
        _format_duration(eta_seconds),
    )


def _median_seconds(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _eta_seconds(
    total_epochs: int, current_epoch: int, step_seconds: Sequence[float]
) -> float:
    """Remaining wall-clock estimate from the rolling median step duration."""
    remaining = max(0, total_epochs - current_epoch)
    return remaining * _median_seconds(step_seconds)


def _format_duration(seconds: float) -> str:
    """Compact human-readable duration: ``12s``, ``38m``, ``2h05m``."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    if hours < 24:
        return f"{hours}h{remaining_minutes:02d}m"
    return f"{hours // 24}d{hours % 24}h"
