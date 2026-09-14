from copy import deepcopy
from pathlib import Path

import pytest

from slow_manifold.config import (
    ConfigError,
    ResolvedExperiment,
    get_config_value,
    resolve_experiment,
)
from slow_manifold.training import TrainConfig

ROOT = Path(__file__).parents[1]
PHASE5_FORMAL_RECIPES = (
    ("phase5_taskA_rank3.yaml", "delayed_interval_categorization", 3, 64),
    ("phase5_taskA_rank5.yaml", "delayed_interval_categorization", 5, 64),
    ("phase5_taskB_rank3.yaml", "delayed_interval_reproduction", 3, 32),
    ("phase5_taskB_rank5.yaml", "delayed_interval_reproduction", 5, 32),
)


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


def test_config_value_paths_can_index_phase_weight_lists() -> None:
    resolved = resolve_experiment(
        ROOT / "experiments" / "phase1_rank2_baseline.yaml"
    )

    assert get_config_value(
        resolved.components, "train.loss.phase_weights.0.lambda"
    ) == pytest.approx(1.0)


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

    assert resolved.components["train"]["epochs"] == 400000
    assert resolved.components["train"]["batch_size"] == 64
    assert resolved.components["train"]["validation_every"] == 10
    assert float(resolved.components["train"]["optimizer"]["learning_rate"]) == 1e-4
    assert resolved.components["train"]["num_threads"] == 2
    assert resolved.components["train"]["loss"] == {
        "name": "phase_normalized_mse",
        "phase_weights": [
            {"phase": "pre_response", "lambda": 1.0},
            {"phase": "response", "lambda": 1.0},
        ],
    }
    assert resolved.components["analysis"]["representative_selection"]["mode"] == "auto"
    assert resolved.components["analysis"]["grid_points"] == 100
    assert resolved.components["analysis"]["square_coordinate_bounds"] is True
    assert [
        trial["interval"]
        for trial in resolved.components["analysis"]["evaluation_trials"]
    ] == [3.0, 4.5, 5.5, 7.0]
    assert resolved.components["visualization"]["loss_y_scale"] == "log"
    assert resolved.components["visualization"]["trajectory_line_width"] == 0.9
    assert resolved.components["train"]["rollout"]["initial_state"]["std"] == 0.1
    assert resolved.components["train"]["rollout"]["neural_noise"]["std"] == 1e-3
    assert resolved.condition_label == "lr-1e-4"


@pytest.mark.parametrize(
    ("recipe_name", "task_name", "rank", "batch_size"),
    PHASE5_FORMAL_RECIPES,
)
def test_phase5_formal_rank_recipes_resolve_a_bounded_static_protocol(
    recipe_name: str,
    task_name: str,
    rank: int,
    batch_size: int,
) -> None:
    resolved = resolve_experiment(ROOT / "experiments" / recipe_name)
    components = resolved.components

    assert resolved.seed == 20260903
    assert components["task"]["name"] == task_name
    assert components["model"]["state_size"] == 64
    assert components["model"]["rank"] == rank
    assert components["train"]["device"] == "cuda"
    assert components["train"]["epochs"] == 400000
    assert components["train"]["batch_size"] == batch_size
    assert components["train"]["validation_every"] == 10
    assert components["train"]["checkpoint"]["every"] == 1000
    assert components["train"]["optimizer"]["learning_rate"] == pytest.approx(1e-4)
    assert components["analysis"]["representative_selection"]["mode"] == "auto"
    assert components["analysis"]["trajectory_count"] == 128
    assert components["visualization"]["render_movies"] is False
    assert resolved.condition_label == f"rank-{rank}__lr-1e-4"

    if task_name == "delayed_interval_reproduction":
        assert components["model"]["dt"] == pytest.approx(1.0)
        assert components["model"]["tau"] == pytest.approx(10.0)
        assert [
            item["phase"] for item in components["train"]["loss"]["phase_weights"]
        ] == ["pre_go", "reproduction_wait", "response"]
        assert [
            item["interval"] for item in components["analysis"]["evaluation_trials"]
        ] == [30.0, 50.0, 75.0, 99.0]


@pytest.mark.parametrize(
    ("rank3_recipe", "rank5_recipe"),
    [
        ("phase5_taskA_rank3.yaml", "phase5_taskA_rank5.yaml"),
        ("phase5_taskB_rank3.yaml", "phase5_taskB_rank5.yaml"),
    ],
)
def test_phase5_rank_pairs_differ_only_in_model_rank(
    rank3_recipe: str, rank5_recipe: str
) -> None:
    rank3 = resolve_experiment(ROOT / "experiments" / rank3_recipe)
    rank5 = resolve_experiment(ROOT / "experiments" / rank5_recipe)

    assert rank3.name == rank5.name
    assert rank3.seed == rank5.seed
    for component in ("task", "train", "analysis", "visualization", "reporting"):
        assert rank3.components[component] == rank5.components[component]
    rank3_model = deepcopy(rank3.components["model"])
    rank5_model = deepcopy(rank5.components["model"])
    assert rank3_model.pop("rank") == 3
    assert rank5_model.pop("rank") == 5
    assert rank3_model == rank5_model


def test_legacy_train_config_resolves_to_zero_noise() -> None:
    resolved = resolve_experiment(
        ROOT / "experiments" / "phase1_rank2_baseline.yaml"
    )
    legacy = deepcopy(resolved.components["train"])
    legacy.pop("rollout")
    legacy.pop("validation_every")
    legacy["device"] = "cpu"

    config = TrainConfig.from_mapping(legacy)

    assert config.rollout.initial_state.std == 0.0
    assert config.rollout.neural_noise.std == 0.0
    assert config.validation_every == 1


def test_train_config_rejects_nonpositive_validation_frequency() -> None:
    resolved = resolve_experiment(
        ROOT / "experiments" / "phase1_rank2_baseline.yaml"
    )
    invalid = deepcopy(resolved.components["train"])
    invalid["device"] = "cpu"
    invalid["validation_every"] = 0

    with pytest.raises(ValueError, match="frequencies must be positive"):
        TrainConfig.from_mapping(invalid)


def test_train_config_rejects_duplicate_loss_phases() -> None:
    resolved = resolve_experiment(
        ROOT / "experiments" / "phase1_rank2_baseline.yaml"
    )
    invalid = deepcopy(resolved.components["train"])
    invalid["device"] = "cpu"
    invalid["loss"]["phase_weights"] = [
        {"phase": "response", "lambda": 1.0},
        {"phase": "response", "lambda": 1.0},
    ]

    with pytest.raises(ValueError, match="unique"):
        TrainConfig.from_mapping(invalid)


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


def test_training_condition_changes_when_rollout_noise_changes() -> None:
    baseline = resolve_experiment(
        ROOT / "experiments" / "phase1_rank2_baseline.yaml"
    )
    components = deepcopy(baseline.components)
    components["train"]["rollout"]["neural_noise"]["std"] = 0.0
    zero_noise = ResolvedExperiment(
        name=baseline.name,
        seed=baseline.seed,
        components=components,
        component_sources=deepcopy(baseline.component_sources),
        source=baseline.source,
        identity_fields=deepcopy(baseline.identity_fields),
    )

    assert zero_noise.condition_fingerprint != baseline.condition_fingerprint
    assert zero_noise.default_run_dir() != baseline.default_run_dir()


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
