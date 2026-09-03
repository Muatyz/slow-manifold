"""Runtime metadata shared by reproducible workflows."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Mapping, Sequence


def collect_runtime_metadata(
    *,
    experiment_name: str,
    seed: int,
    rng_streams: Mapping[str, str],
    packages: Sequence[str] = ("numpy", "matplotlib", "pyyaml"),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect environment, package, RNG, and Git provenance."""
    git_commit, git_dirty = _git_state()
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_name,
        "seed": seed,
        "rng_streams": dict(rng_streams),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in packages},
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _git_state() -> tuple[str | None, bool | None]:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None

    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    return commit, bool(status)
