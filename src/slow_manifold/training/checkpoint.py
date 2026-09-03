"""Atomic PyTorch checkpoint I/O."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import torch


def save_checkpoint(payload: dict, path: str | Path) -> Path:
    """Atomically write a checkpoint without exposing a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    """Load a trusted project checkpoint."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)
