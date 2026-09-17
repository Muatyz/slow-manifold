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
    slow_point_epochs: Sequence[int] | None = None,
) -> str:
    """Identify the inputs that determine structured latent diagnostics."""
    latent_analysis = deepcopy(dict(analysis))
    # Selection rules are presentation concerns, so they stay out of the
    # generic analysis mapping. For rank >= 3, the concrete selected epochs
    # are added below because they determine where full-state refinement runs.
    latent_analysis.pop("representative_selection", None)
    latent_analysis.pop("representative_epochs", None)
    payload: dict[str, Any] = {
        "task": task,
        "model": model,
        "analysis": latent_analysis,
        "checkpoint_epochs": [int(epoch) for epoch in checkpoint_epochs],
    }
    search = latent_analysis.get("trajectory_slow_point_search", {})
    search_enabled = not isinstance(search, Mapping) or search.get("enabled", True)
    if int(model.get("rank", 2)) >= 3 and search_enabled:
        payload["slow_point_epochs"] = [
            int(epoch) for epoch in (slow_point_epochs or ())
        ]
    return _fingerprint(payload)


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
    slow_point_epochs: Sequence[int] | None = None,
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
            "slow_point_epochs": [
                int(epoch) for epoch in (slow_point_epochs or ())
            ],
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
