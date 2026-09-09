"""Config-driven, human-readable experiment startup summaries."""

from __future__ import annotations

from typing import Any, Mapping

from slow_manifold.config import ConfigError, get_config_value


def startup_summary_lines(
    components: Mapping[str, Mapping[str, Any]],
    reporting: Mapping[str, Any],
) -> tuple[str, ...]:
    """Render configured component values without duplicating their values."""
    enabled = reporting.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("reporting.enabled must be true or false")
    if not enabled:
        return ()
    sections = reporting.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ConfigError("reporting.sections must be a non-empty list")

    lines: list[str] = []
    for section in sections:
        if not isinstance(section, Mapping):
            raise ConfigError("Each reporting section must be a mapping")
        label = section.get("label")
        title = section.get("title")
        fields = section.get("fields", [])
        if not isinstance(label, str) or not label.strip():
            raise ConfigError("Each reporting section requires a label")
        if title is not None and not isinstance(title, str):
            raise ConfigError("reporting section title must be a string or null")
        if not isinstance(fields, list):
            raise ConfigError("reporting section fields must be a list")

        values = [title] if title else []
        for field in fields:
            if not isinstance(field, Mapping):
                raise ConfigError("Each reporting field must be a mapping")
            field_label = field.get("label")
            path = field.get("path")
            if not isinstance(field_label, str) or not field_label.strip():
                raise ConfigError("Each reporting field requires a label")
            if not isinstance(path, str) or not path.strip():
                raise ConfigError("Each reporting field requires a config path")
            value = get_config_value(components, path)
            if isinstance(value, (Mapping, list, tuple)):
                raise ConfigError(
                    f"Reporting path '{path}' must resolve to a scalar value"
                )
            if isinstance(value, float):
                rendered = f"{value:.8g}"
            else:
                rendered = str(value)
            values.append(f"{field_label}={rendered}")
        lines.append(f"{label}: " + " | ".join(values))
    return tuple(lines)
