"""Project-level logging: one human-readable event log per run.

Every workflow writes to the same logger, which echoes each record to the
console and appends it to ``run.log`` in the run directory.  Records carry a
``phase`` tag (e.g. ``train``, ``analysis``) so an agent or reader can follow
which stage produced an event, and events are kept human-readable:

    ``2026-09-03 21:10:03 INFO [train] step=100 train_loss=... eta=38m``

Timestamps in ``run.log`` are local wall-clock time for readability; machine
consumers should prefer the UTC fields in ``status.yaml``/``metadata.yaml``.

Modules configure nothing themselves: they only call :func:`get_logger` and
record through the returned adapter.  ``setup_run_logging`` is invoked once by
the workflow that owns the run directory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

LOGGER_NAME = "slow_manifold"
RUN_LOG_NAME = "run.log"
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(phase)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _DefaultPhaseFilter(logging.Filter):
    """Stamp records that did not carry a phase with a neutral placeholder."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "phase"):
            record.phase = "-"
        return True


class _PhaseAdapter(logging.LoggerAdapter):
    """Attach a fixed phase tag to every record logged through the adapter."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        extra = kwargs.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            kwargs["extra"] = extra
        extra.setdefault("phase", self.extra["phase"])
        return msg, kwargs


def _logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_run_logging(
    run_dir: str | Path, *, level: int = logging.INFO
) -> logging.Logger:
    """Configure the project logger for one run directory (idempotent).

    Writes human-readable records to ``run_dir/run.log`` and mirrors them to
    stdout.  Re-invocation replaces previous handlers, so repeated runs inside
    one process (e.g. tests) never accumulate duplicate output.
    """
    logger = _logger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(level)
    logger.propagate = False
    logger.addFilter(_DefaultPhaseFilter())

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(destination / RUN_LOG_NAME, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_logger(phase: str = "-") -> logging.LoggerAdapter:
    """Return an adapter whose records carry the given ``phase`` tag.

    Safe to call before :func:`setup_run_logging`: until a run directory is
    configured the records are dropped instead of raising, which keeps library
    imports side-effect free.
    """
    if not _logger().handlers:
        _logger().addHandler(logging.NullHandler())
    return _PhaseAdapter(_logger(), {"phase": phase})
