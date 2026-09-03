"""Training loop for configured rank-2 CTRNN experiments."""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from slow_manifold.models import Rank2CTRNN
from slow_manifold.tasks import IntervalCategorizationTask, TaskBatch
from slow_manifold.training.checkpoint import save_checkpoint
from slow_manifold.utils.logging import get_logger


class TrainConfigError(ValueError):
    """Raised when training configuration is invalid."""


@dataclass(frozen=True)
class TrainConfig:
    name: str
    device: str
    epochs: int
    batch_size: int
    validation_batch_size: int
    log_every: int
    optimizer_name: str
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    gradient_clip_norm: float | None
    checkpoint_every: int
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
                log_every=int(data["log_every"]),
                optimizer_name=str(optimizer["name"]),
                learning_rate=float(optimizer["learning_rate"]),
                betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
                weight_decay=float(optimizer["weight_decay"]),
                gradient_clip_norm=None if clip is None else float(clip),
                checkpoint_every=int(checkpoint["every"]),
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
        if self.epochs <= 0 or self.log_every <= 0 or self.checkpoint_every <= 0:
            raise TrainConfigError("epochs and frequencies must be positive")
        if self.batch_size <= 0 or self.batch_size % 2:
            raise TrainConfigError("batch_size must be a positive even integer")
        if self.validation_batch_size <= 0 or self.validation_batch_size % 2:
            raise TrainConfigError(
                "validation_batch_size must be a positive even integer"
            )
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
    history: tuple[dict[str, float | int], ...]
    checkpoint_paths: dict[int, Path]
    best_epoch: int


def train_model(
    *,
    model: Rank2CTRNN,
    train_task: IntervalCategorizationTask,
    validation_batch: TaskBatch,
    train_rng: np.random.Generator,
    config: TrainConfig,
    run_dir: Path,
    snapshot_epochs: Sequence[int],
    resolved_config: Mapping[str, Any],
) -> TrainingResult:
    """Train while recording Dinc-style optimization diagnostics."""
    if config.num_threads is not None:
        # Cap the PyTorch intra-op thread pool per the resolved train config.
        # Wall-clock is insensitive to this value on CUDA (benchmarked), while a
        # lower cap frees CPU cores for concurrent multi-seed runs.  This only
        # touches the current process; interop threads and global defaults are
        # left untouched so the rest of the session is unaffected.
        torch.set_num_threads(config.num_threads)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    requested_snapshots = {int(epoch) for epoch in snapshot_epochs}
    requested_snapshots.update({0, config.epochs})
    checkpoint_paths: dict[int, Path] = {}
    history: list[dict[str, float | int]] = []
    metrics_path = run_dir / "metrics.csv"
    logger = get_logger("train")
    logger.info(
        "training started epochs=%d batch_size=%d device=%s learning_rate=%s "
        "torch_intra_threads=%d",
        config.epochs,
        config.batch_size,
        config.device,
        config.learning_rate,
        torch.get_num_threads(),
    )
    training_started = time.perf_counter()
    elapsed_seconds = 0.0
    step_seconds: deque[float] = deque(maxlen=100)

    def _measured_step(
        *, batch: TaskBatch, epoch: int, update: bool
    ) -> dict[str, float | int]:
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
        )
        duration = time.perf_counter() - start
        elapsed_seconds += duration
        row["epoch_seconds"] = duration
        row["elapsed_seconds"] = elapsed_seconds
        return row

    initial_batch = train_task.generate_batch(config.batch_size, train_rng)
    initial_row = _measured_step(batch=initial_batch, epoch=0, update=False)
    history.append(initial_row)
    _append_metric(metrics_path, initial_row, create=True)
    checkpoint_paths[0] = _save_epoch_checkpoint(
        model, optimizer, train_rng, config, history, resolved_config, run_dir, 0
    )
    save_checkpoint(
        _checkpoint_payload(
            model, optimizer, train_rng, config, history, resolved_config, 0
        ),
        run_dir / "checkpoints" / "initial.pt",
    )

    best_epoch = 0
    best_loss = float(initial_row["validation_loss"])
    save_checkpoint(
        _checkpoint_payload(
            model, optimizer, train_rng, config, history, resolved_config, 0
        ),
        run_dir / "checkpoints" / "best.pt",
    )

    for epoch in range(1, config.epochs + 1):
        batch = train_task.generate_batch(config.batch_size, train_rng)
        row = _measured_step(batch=batch, epoch=epoch, update=True)
        step_seconds.append(row["epoch_seconds"])
        history.append(row)
        _append_metric(metrics_path, row, create=False)

        if float(row["validation_loss"]) < best_loss:
            best_loss = float(row["validation_loss"])
            best_epoch = epoch
            save_checkpoint(
                _checkpoint_payload(
                    model,
                    optimizer,
                    train_rng,
                    config,
                    history,
                    resolved_config,
                    epoch,
                ),
                run_dir / "checkpoints" / "best.pt",
            )

        should_snapshot = (
            epoch % config.checkpoint_every == 0
            or epoch in requested_snapshots
            or epoch == config.epochs
        )
        if should_snapshot:
            start = time.perf_counter()
            checkpoint_paths[epoch] = _save_epoch_checkpoint(
                model,
                optimizer,
                train_rng,
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
            logger.info(
                "step=%d train_loss=%.6g validation_loss=%.6g accuracy=%.3f "
                "epoch_seconds=%.2fs elapsed=%.1fs eta=%s",
                epoch,
                row["train_loss"],
                row["validation_loss"],
                row["validation_accuracy"],
                row["epoch_seconds"],
                row["elapsed_seconds"],
                _format_duration(
                    _eta_seconds(config.epochs, epoch, step_seconds)
                ),
            )

    final_payload = _checkpoint_payload(
        model,
        optimizer,
        train_rng,
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
) -> dict[str, float | int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train = _torch_batch(batch, device, model.config.torch_dtype)
    prediction, _, _ = model.rollout(train["inputs"])
    train_loss = _masked_mse(prediction, train["target"], train["loss_mask"])
    train_loss.backward()
    recurrent_gradient_norm = _recurrent_gradient_norm(model)
    total_gradient_norm = _total_gradient_norm(model)
    if update:
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        validation = _torch_batch(
            validation_batch, device, model.config.torch_dtype
        )
        validation_prediction, _, _ = model.rollout(validation["inputs"])
        validation_loss = _masked_mse(
            validation_prediction, validation["target"], validation["loss_mask"]
        )
        validation_accuracy = _response_accuracy(
            validation_prediction, validation["target"]
        )
        parameter_diagnostics = _parameter_diagnostics(model)
    return {
        "epoch": epoch,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "train_loss": float(train_loss.detach().cpu()),
        "validation_loss": float(validation_loss.cpu()),
        "validation_accuracy": float(validation_accuracy.cpu()),
        "recurrent_gradient_norm": recurrent_gradient_norm,
        "total_gradient_norm": total_gradient_norm,
        **parameter_diagnostics,
    }


def _torch_batch(
    batch: TaskBatch, device: torch.device, dtype: torch.dtype
) -> dict[str, Tensor]:
    return {
        "inputs": torch.as_tensor(batch.inputs, dtype=dtype, device=device),
        "target": torch.as_tensor(batch.target, dtype=dtype, device=device),
        "loss_mask": torch.as_tensor(batch.loss_mask, dtype=dtype, device=device),
    }


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return torch.sum(mask * (prediction - target).square()) / torch.sum(mask)


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
    return {
        "m_norm": float(torch.linalg.vector_norm(model.m).cpu()),
        "n_norm": float(torch.linalg.vector_norm(model.n).cpu()),
        "w_norm": float(torch.linalg.vector_norm(singular_values).cpu()),
        "singular_value_1": float(singular_values[0].cpu()),
        "singular_value_2": float(singular_values[1].cpu()),
    }


def _checkpoint_payload(
    model: Rank2CTRNN,
    optimizer: torch.optim.Optimizer,
    train_rng: np.random.Generator,
    config: TrainConfig,
    history: Sequence[dict[str, float | int]],
    resolved_config: Mapping[str, Any],
    epoch: int,
) -> dict:
    return {
        "format_version": 1,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_config": config.__dict__,
        "resolved_config": dict(resolved_config),
        "metrics": dict(history[-1]),
        "rng_state": {
            "task_train": train_rng.bit_generator.state,
            "torch_cpu": torch.random.get_rng_state(),
        },
    }


def _save_epoch_checkpoint(
    model: Rank2CTRNN,
    optimizer: torch.optim.Optimizer,
    train_rng: np.random.Generator,
    config: TrainConfig,
    history: Sequence[dict[str, float | int]],
    resolved_config: Mapping[str, Any],
    run_dir: Path,
    epoch: int,
) -> Path:
    return save_checkpoint(
        _checkpoint_payload(
            model,
            optimizer,
            train_rng,
            config,
            history,
            resolved_config,
            epoch,
        ),
        run_dir / "checkpoints" / f"epoch-{epoch:06d}.pt",
    )


def _append_metric(
    path: Path, row: Mapping[str, float | int], *, create: bool
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
