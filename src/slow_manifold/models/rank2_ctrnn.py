"""Rank-constrained continuous-time recurrent neural network."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping

import torch
from torch import Tensor, nn


class ModelConfigError(ValueError):
    """Raised when the rank-2 CTRNN configuration is invalid."""


@dataclass(frozen=True)
class Rank2CTRNNConfig:
    name: str
    input_size: int
    state_size: int
    rank: int
    output_size: int
    tau: float
    dt: float
    dtype: str
    factor_scale: float
    input_scale: float
    bias_scale: float
    readout_scale: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Rank2CTRNNConfig":
        initialization = data.get("initialization")
        if not isinstance(initialization, Mapping):
            raise ModelConfigError("model.initialization must be a mapping")
        try:
            config = cls(
                name=str(data["name"]),
                input_size=int(data["input_size"]),
                state_size=int(data["state_size"]),
                rank=int(data["rank"]),
                output_size=int(data["output_size"]),
                tau=float(data["tau"]),
                dt=float(data["dt"]),
                dtype=str(data["dtype"]),
                factor_scale=float(initialization["factor_scale"]),
                input_scale=float(initialization["input_scale"]),
                bias_scale=float(initialization["bias_scale"]),
                readout_scale=float(initialization["readout_scale"]),
            )
        except KeyError as error:
            raise ModelConfigError(f"Missing model field: {error.args[0]}") from error
        config.validate(data)
        return config

    def validate(self, raw: Mapping[str, Any]) -> None:
        if raw.get("activation") != "tanh":
            raise ModelConfigError("Only tanh activation is supported")
        if raw.get("integrator") != "euler":
            raise ModelConfigError("Only Euler integration is supported")
        if self.rank != 2:
            raise ModelConfigError("This model requires rank=2")
        if min(self.input_size, self.state_size, self.output_size) <= 0:
            raise ModelConfigError("Model dimensions must be positive")
        if self.rank > self.state_size:
            raise ModelConfigError("rank must not exceed state_size")
        if self.tau <= 0 or self.dt <= 0 or self.dt > self.tau:
            raise ModelConfigError("Require 0 < dt <= tau")
        if self.dtype not in {"float32", "float64"}:
            raise ModelConfigError("dtype must be float32 or float64")
        scales = (
            self.factor_scale,
            self.input_scale,
            self.bias_scale,
            self.readout_scale,
        )
        if any(scale < 0 for scale in scales):
            raise ModelConfigError("Initialization scales must be non-negative")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == "float32" else torch.float64


class Rank2CTRNN(nn.Module):
    """Vanilla CTRNN with recurrent matrix ``W = M N^T`` and latent readout."""

    def __init__(
        self, config: Rank2CTRNNConfig, *, generator: torch.Generator | None = None
    ) -> None:
        super().__init__()
        self.config = config
        factory = {"dtype": config.torch_dtype}
        self.m = nn.Parameter(torch.empty(config.state_size, config.rank, **factory))
        self.n = nn.Parameter(torch.empty(config.state_size, config.rank, **factory))
        self.input_weight = nn.Parameter(
            torch.empty(config.state_size, config.input_size, **factory)
        )
        self.bias = nn.Parameter(torch.empty(config.state_size, **factory))
        self.readout_weight = nn.Parameter(
            torch.empty(config.rank, config.output_size, **factory)
        )
        self.readout_bias = nn.Parameter(torch.zeros(config.output_size, **factory))
        self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        cfg = self.config
        factor_std = cfg.factor_scale / sqrt(cfg.state_size)
        input_std = cfg.input_scale / sqrt(cfg.input_size)
        readout_std = cfg.readout_scale / sqrt(cfg.rank)
        with torch.no_grad():
            self.m.normal_(0.0, factor_std, generator=generator)
            self.n.normal_(0.0, factor_std, generator=generator)
            self.input_weight.normal_(0.0, input_std, generator=generator)
            self.bias.normal_(0.0, cfg.bias_scale, generator=generator)
            self.readout_weight.normal_(0.0, readout_std, generator=generator)
            self.readout_bias.zero_()

    @property
    def recurrent_matrix(self) -> Tensor:
        return self.m @ self.n.T

    @property
    def effective_readout(self) -> Tensor:
        """Readout weights in full state coordinates."""
        return self.n @ self.readout_weight

    def latent(self, state: Tensor) -> Tensor:
        return state @ self.n

    def flow(self, state: Tensor, inputs: Tensor) -> Tensor:
        kappa = self.latent(state)
        drive = kappa @ self.m.T + inputs @ self.input_weight.T + self.bias
        return (-state + torch.tanh(drive)) / self.config.tau

    def latent_flow(self, kappa: Tensor, inputs: Tensor) -> Tensor:
        """Exact closed latent flow for ``kappa = N^T x``."""
        drive = kappa @ self.m.T + inputs @ self.input_weight.T + self.bias
        return (-kappa + torch.tanh(drive) @ self.n) / self.config.tau

    def row_space_flow(
        self, coordinates: Tensor, inputs: Tensor, basis: Tensor
    ) -> Tensor:
        """Exact flow in an orthonormal basis spanning the row space of ``W``."""
        loading = self.recurrent_matrix @ basis
        drive = (
            coordinates @ loading.T
            + inputs @ self.input_weight.T
            + self.bias
        )
        return (-coordinates + torch.tanh(drive) @ basis) / self.config.tau

    def row_space_jacobian(
        self, coordinates: Tensor, inputs: Tensor, basis: Tensor
    ) -> Tensor:
        """Analytic continuous-time Jacobian of :meth:`row_space_flow`."""
        loading = self.recurrent_matrix @ basis
        drive = (
            coordinates @ loading.T
            + inputs @ self.input_weight.T
            + self.bias
        )
        activation_slope = 1.0 - torch.tanh(drive).square()
        recurrent_term = torch.einsum(
            "hi,...h,hj->...ij", basis, activation_slope, loading
        )
        identity = torch.eye(
            basis.shape[1], dtype=coordinates.dtype, device=coordinates.device
        )
        return (recurrent_term - identity) / self.config.tau

    def readout(self, kappa: Tensor) -> Tensor:
        return torch.tanh(kappa @ self.readout_weight + self.readout_bias)

    def rollout(
        self, inputs: Tensor, initial_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Euler-integrate a batch and return output, full state, and latent state."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.config.input_size:
            raise ValueError("inputs must have shape [batch, time, input_size]")
        batch_size, steps, _ = inputs.shape
        state = (
            torch.zeros(
                batch_size,
                self.config.state_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            if initial_state is None
            else initial_state
        )
        outputs: list[Tensor] = []
        states: list[Tensor] = []
        latents: list[Tensor] = []
        for step in range(steps):
            state = state + self.config.dt * self.flow(state, inputs[:, step])
            kappa = self.latent(state)
            states.append(state)
            latents.append(kappa)
            outputs.append(self.readout(kappa))
        return (
            torch.stack(outputs, dim=1),
            torch.stack(states, dim=1),
            torch.stack(latents, dim=1),
        )
