"""Resolved configuration records for post-training workflow stages."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from slow_manifold.config import ConfigError, dump_yaml, load_yaml

_SCHEMA_VERSION = 1


def analysis_fingerprint(
    *,
    task: Mapping[str, Any],
    model: Mapping[str, Any],
    analysis: Mapping[str, Any],
    checkpoint_epochs: Sequence[int],
) -> str:
    """Identify the inputs that determine structured latent diagnostics."""
    latent_analysis = deepcopy(dict(analysis))
    # Representative checkpoint selection consumes metrics and existing
    # checkpoints; it does not change trajectories, fields, or Jacobians.
    # Keeping it out of this fingerprint avoids an expensive grid recompute
    # when a user only changes the selected presentation epochs.
    latent_analysis.pop("representative_selection", None)
    latent_analysis.pop("representative_epochs", None)
    return _fingerprint(
        {
            "task": task,
            "model": model,
            "analysis": latent_analysis,
            "checkpoint_epochs": [int(epoch) for epoch in checkpoint_epochs],
        }
    )


def visualization_fingerprint(
    *,
    analysis_fingerprint_value: str,
    visualization: Mapping[str, Any],
    representative_epochs: Sequence[int],
) -> str:
    """Identify the diagnostics and display choices used by a figure set."""
    return _fingerprint(
        {
            "analysis_fingerprint": analysis_fingerprint_value,
            "visualization": visualization,
            "representative_epochs": [
                int(epoch) for epoch in representative_epochs
            ],
        }
    )


def write_analysis_stage_config(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    fingerprint: str,
    checkpoint_epochs: Sequence[int],
    config_source: str,
    source_experiment: str | None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Write the complete resolved analysis component used successfully."""
    dump_yaml(
        {
            "schema_version": _SCHEMA_VERSION,
            "kind": "analysis",
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_source": config_source,
            "source_experiment": source_experiment,
            "fingerprint": fingerprint,
            "checkpoint_epochs": [int(epoch) for epoch in checkpoint_epochs],
            "cli_overrides": deepcopy(dict(cli_overrides or {})),
            "config": deepcopy(dict(config)),
        },
        path,
    )


def write_visualization_stage_config(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    fingerprint: str,
    analysis_fingerprint_value: str,
    representative_epochs: Sequence[int],
    config_source: str,
    source_experiment: str | None,
    cli_overrides: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """Write the complete resolved visualization component used successfully."""
    dump_yaml(
        {
            "schema_version": _SCHEMA_VERSION,
            "kind": "visualization",
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_source": config_source,
            "source_experiment": source_experiment,
            "output_dir": str(Path(output_dir).resolve()) if output_dir else None,
            "fingerprint": fingerprint,
            "analysis_fingerprint": analysis_fingerprint_value,
            "representative_epochs": [
                int(epoch) for epoch in representative_epochs
            ],
            "cli_overrides": deepcopy(dict(cli_overrides or {})),
            "config": deepcopy(dict(config)),
        },
        path,
    )


def load_stage_component(
    path: str | Path, *, expected_kind: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one complete stage component and its provenance record."""
    record = load_yaml(path)
    if record.get("kind") != expected_kind:
        raise ConfigError(
            f"Expected a {expected_kind!r} stage config in {Path(path).resolve()}"
        )
    config = record.get("config")
    if not isinstance(config, Mapping):
        raise ConfigError(
            f"Stage config has no resolved 'config' mapping: {Path(path).resolve()}"
        )
    return deepcopy(dict(config)), record


def _fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]
