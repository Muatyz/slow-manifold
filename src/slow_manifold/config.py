"""Configuration loading and experiment composition."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration cannot be resolved safely."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a YAML mapping in {config_path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class ResolvedExperiment:
    """A recipe with all referenced component values materialized."""

    name: str
    seed: int
    components: dict[str, dict[str, Any]]
    component_sources: dict[str, str]
    source: str
    identity_fields: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "seed": self.seed,
            "source": self.source,
            "component_sources": deepcopy(self.component_sources),
            "components": deepcopy(self.components),
        }
        if self.identity_fields:
            data["identity"] = {"label_fields": deepcopy(self.identity_fields)}
        return data

    @property
    def condition_fingerprint(self) -> str:
        """Stable hash of parameters that can change the trained network."""
        payload = {
            name: self.components[name]
            for name in ("task", "model", "train")
            if name in self.components
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:10]

    @property
    def condition_label(self) -> str | None:
        """Human-readable label derived from real resolved config values."""
        if not self.identity_fields:
            return None
        parts = []
        for label, path in self.identity_fields.items():
            value = get_config_value(self.components, path)
            parts.append(f"{_slug(label)}-{_slug(value)}")
        return "__".join(parts)

    def default_run_dir(self, runs_root: str | Path = "runs") -> Path:
        """Derive a traceable run path from condition identity and seed."""
        root = Path(runs_root) / self.name
        fingerprint = f"cfg-{self.condition_fingerprint}"
        condition = (
            fingerprint
            if self.condition_label is None
            else f"{self.condition_label}--{fingerprint}"
        )
        return (root / condition / f"seed-{self.seed}").resolve()


def resolve_experiment(path: str | Path) -> ResolvedExperiment:
    """Resolve component references and sparse per-component overrides."""
    recipe_path = Path(path).resolve()
    recipe = load_yaml(recipe_path)

    name = recipe.get("name")
    seed = recipe.get("seed")
    references = recipe.get("components")
    overrides = recipe.get("overrides", {})
    identity = recipe.get("identity", {})
    if "tag" in recipe:
        raise ConfigError(
            "Experiment 'tag' is no longer supported; encode the condition in "
            "component overrides and optional identity.label_fields"
        )
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("Experiment 'name' must be a non-empty string")
    _require_path_component(name, "name")
    if not isinstance(seed, int):
        raise ConfigError("Experiment 'seed' must be an integer")
    if not isinstance(identity, Mapping):
        raise ConfigError("Experiment 'identity' must be a mapping")
    unknown_identity_fields = set(identity) - {"label_fields"}
    if unknown_identity_fields:
        names = ", ".join(sorted(str(name) for name in unknown_identity_fields))
        raise ConfigError(f"Unknown experiment identity fields: {names}")
    label_fields = identity.get("label_fields", {})
    if not isinstance(label_fields, Mapping):
        raise ConfigError("Experiment identity.label_fields must be a mapping")
    if not isinstance(references, Mapping) or not references:
        raise ConfigError("Experiment 'components' must be a non-empty mapping")
    if not isinstance(overrides, Mapping):
        raise ConfigError("Experiment 'overrides' must be a mapping")

    unknown_overrides = set(overrides) - set(references)
    if unknown_overrides:
        names = ", ".join(sorted(unknown_overrides))
        raise ConfigError(f"Overrides refer to unknown components: {names}")

    components: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for component_name, reference in references.items():
        if not isinstance(component_name, str) or not isinstance(reference, str):
            raise ConfigError("Component names and paths must be strings")
        component_path = (recipe_path.parent / reference).resolve()
        component = load_yaml(component_path)
        component_override = overrides.get(component_name, {})
        if not isinstance(component_override, Mapping):
            raise ConfigError(f"Override for '{component_name}' must be a mapping")
        components[component_name] = deep_merge(component, component_override)
        sources[component_name] = reference

    resolved_identity_fields: dict[str, str] = {}
    for label, value_path in label_fields.items():
        if not isinstance(label, str) or not label.strip():
            raise ConfigError("Identity labels must be non-empty strings")
        if not isinstance(value_path, str) or not value_path.strip():
            raise ConfigError("Identity value paths must be non-empty strings")
        value = get_config_value(components, value_path)
        if isinstance(value, (Mapping, list, tuple)):
            raise ConfigError(
                f"Identity path '{value_path}' must resolve to a scalar value"
            )
        resolved_identity_fields[label] = value_path

    return ResolvedExperiment(
        name=name,
        seed=seed,
        components=components,
        component_sources=sources,
        source=str(recipe_path),
        identity_fields=resolved_identity_fields,
    )


def get_config_value(components: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path such as ``train.optimizer.learning_rate``."""
    value: Any = components
    for part in path.split("."):
        if not part:
            raise ConfigError(f"Config value path does not exist: {path}")
        if isinstance(value, Mapping) and part in value:
            value = value[part]
            continue
        if isinstance(value, (list, tuple)) and part.isdigit():
            index = int(part)
            if 0 <= index < len(value):
                value = value[index]
                continue
        raise ConfigError(f"Config value path does not exist: {path}")
    return value


def _require_path_component(value: str, field: str) -> None:
    """Reject values that could escape or collide inside a run directory."""
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ConfigError(f"Experiment '{field}' must be a single path-safe name")


def _slug(value: Any) -> str:
    """Format a scalar as a compact, path-safe condition token."""
    if isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, float):
        magnitude = abs(value)
        if value != 0 and (magnitude < 0.01 or magnitude >= 10000):
            mantissa, exponent = f"{value:.6e}".split("e")
            text = f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
        else:
            text = f"{value:.8g}"
    else:
        text = str(value).strip()
    token = re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-.")
    if not token:
        raise ConfigError(f"Cannot form a path-safe identity token from {value!r}")
    return token


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Write a YAML mapping with stable, human-readable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(dict(data), stream, sort_keys=False, allow_unicode=True)
