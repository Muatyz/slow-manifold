from copy import deepcopy
from pathlib import Path

import pytest

from slow_manifold.config import ConfigError, ResolvedExperiment, resolve_experiment

ROOT = Path(__file__).parents[1]


def test_resolve_experiment_applies_sparse_overrides(tmp_path: Path) -> None:
    component = tmp_path / "component.yaml"
    component.write_text("outer:\n  value: 1\n  keep: 2\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "name: example\n"
        "seed: 7\n"
        "components:\n"
        "  task: component.yaml\n"
        "overrides:\n"
        "  task:\n"
        "    outer:\n"
        "      value: 3\n",
        encoding="utf-8",
    )

    resolved = resolve_experiment(recipe)

    assert resolved.components["task"] == {"outer": {"value": 3, "keep": 2}}
    assert resolved.seed == 7
    assert resolved.default_run_dir(tmp_path / "runs") == (
        tmp_path
        / "runs"
        / "example"
        / f"cfg-{resolved.condition_fingerprint}"
        / "seed-7"
    ).resolve()


def test_resolve_experiment_rejects_unknown_override(tmp_path: Path) -> None:
    component = tmp_path / "component.yaml"
    component.write_text("value: 1\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "name: example\n"
        "seed: 7\n"
        "components:\n"
        "  task: component.yaml\n"
        "overrides:\n"
        "  model: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown components"):
        resolve_experiment(recipe)


def test_resolve_experiment_rejects_name_with_path_separator(tmp_path: Path) -> None:
    component = tmp_path / "component.yaml"
    component.write_text("value: 1\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "name: ../outside\n"
        "seed: 7\n"
        "components:\n"
        "  task: component.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="path-safe"):
        resolve_experiment(recipe)


def test_phase1_baseline_resolves_requested_training_schedule() -> None:
    resolved = resolve_experiment(
        ROOT / "experiments" / "phase1_rank2_baseline.yaml"
    )

    assert resolved.components["train"]["epochs"] == 200000
    assert resolved.components["train"]["batch_size"] == 64
    assert float(resolved.components["train"]["optimizer"]["learning_rate"]) == 1e-4
    assert resolved.components["train"]["num_threads"] == 2
    assert resolved.components["analysis"]["representative_epochs"][-1] == 200000
    assert resolved.components["analysis"]["grid_points"] == 41
    assert resolved.condition_label == "lr-1e-4"


def test_training_condition_changes_when_learning_rate_changes() -> None:
    baseline = resolve_experiment(
        ROOT / "experiments" / "phase1_rank2_baseline.yaml"
    )
    components = deepcopy(baseline.components)
    components["train"]["optimizer"]["learning_rate"] = 1.0e-3
    changed_lr = ResolvedExperiment(
        name=baseline.name,
        seed=baseline.seed,
        components=components,
        component_sources=deepcopy(baseline.component_sources),
        source=baseline.source,
        identity_fields=deepcopy(baseline.identity_fields),
    )

    assert changed_lr.condition_label == "lr-1e-3"
    assert changed_lr.condition_fingerprint != baseline.condition_fingerprint
    assert changed_lr.default_run_dir() != baseline.default_run_dir()


def test_legacy_tag_is_rejected(tmp_path: Path) -> None:
    component = tmp_path / "component.yaml"
    component.write_text("value: 1\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "name: example\n"
        "seed: 7\n"
        "tag: lr1e-4\n"
        "components:\n"
        "  task: component.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="no longer supported"):
        resolve_experiment(recipe)
