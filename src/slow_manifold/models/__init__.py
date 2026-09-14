"""Recurrent network models."""

from .rank2_ctrnn import Rank2CTRNN, Rank2CTRNNConfig

LowRankCTRNN = Rank2CTRNN
LowRankCTRNNConfig = Rank2CTRNNConfig

__all__ = [
    "LowRankCTRNN",
    "LowRankCTRNNConfig",
    "Rank2CTRNN",
    "Rank2CTRNNConfig",
]
