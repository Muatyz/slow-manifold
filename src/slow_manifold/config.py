"""Configuration loading and experiment composition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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
    tag: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "seed": self.seed,
            "source": self.source,
            "component_sources": deepcopy(self.component_sources),
            "components": deepcopy(self.components),
        }
        if self.tag is not None:
            data["tag"] = self.tag
        return data

    def default_run_dir(self, runs_root: str | Path = "runs") -> Path:
        """Derive the conventional run directory from experiment identity.

        Runs are separated by an optional ``tag`` so that identical recipes
        with different hyper-parameters (e.g. a learning-rate scan) never
        collide: ``runs/<name>/<tag>-seed-<seed>`` without a tag falls back
        to ``runs/<name>/seed-<seed>``.
        """
        root = Path(runs_root) / self.name
        directory = root / f"seed-{self.seed}"
        if self.tag is not None:
            directory = root / f"{self.tag}-seed-{self.seed}"
        return directory.resolve()


def resolve_experiment(path: str | Path) -> ResolvedExperiment:
    """Resolve component references and sparse per-component overrides."""
    recipe_path = Path(path).resolve()
    recipe = load_yaml(recipe_path)

    name = recipe.get("name")
    seed = recipe.get("seed")
    references = recipe.get("components")
    overrides = recipe.get("overrides", {})
    tag = recipe.get("tag")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("Experiment 'name' must be a non-empty string")
    _require_path_component(name, "name")
    if not isinstance(seed, int):
        raise ConfigError("Experiment 'seed' must be an integer")
    if tag is not None:
        if not isinstance(tag, str) or not tag.strip():
            raise ConfigError("Experiment 'tag' must be a non-empty string")
        _require_path_component(tag, "tag")
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

    return ResolvedExperiment(
        name=name,
        seed=seed,
        tag=tag,
        components=components,
        component_sources=sources,
        source=str(recipe_path),
    )


def _require_path_component(value: str, field: str) -> None:
    """Reject values that could escape or collide inside a run directory."""
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ConfigError(f"Experiment '{field}' must be a single path-safe name")


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Write a YAML mapping with stable, human-readable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(dict(data), stream, sort_keys=False, allow_unicode=True)
