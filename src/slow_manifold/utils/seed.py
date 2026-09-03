"""Deterministic, independent random-number streams."""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(seed: int, namespace: str) -> int:
    """Derive a stable non-negative 63-bit seed from a base seed and name."""
    payload = f"{seed}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def make_rng(seed: int, namespace: str) -> np.random.Generator:
    """Create a stable RNG stream derived from a base seed and namespace."""
    return np.random.default_rng(derive_seed(seed, namespace))
