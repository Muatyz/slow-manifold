from pathlib import Path

import pytest

from slow_manifold.sanity import create_task_sanity_run
from slow_manifold.config import resolve_experiment

ROOT = Path(__file__).parents[1]


def test_task_sanity_workflow_writes_traceable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment_path = ROOT / "experiments" / "phase0_task_sanity.yaml"
    output_dir = resolve_experiment(experiment_path).default_run_dir()

    result = create_task_sanity_run(
        experiment_path,
        batch_size=4,
    )

    assert result == output_dir.resolve()
    assert (result / "config.yaml").is_file()
    assert (result / "metadata.yaml").is_file()
    assert (result / "trial_metadata.yaml").is_file()
    assert (result / "batch.npz").is_file()
    assert (result / "figures" / "task_trials.png").is_file()


def test_task_sanity_workflow_refuses_to_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        create_task_sanity_run(
            ROOT / "experiments" / "phase0_task_sanity.yaml",
            output_dir,
            batch_size=4,
        )


def test_reproduction_task_uses_the_same_sanity_workflow(tmp_path: Path) -> None:
    output_dir = tmp_path / "reproduction-sanity"
    result = create_task_sanity_run(
        ROOT / "experiments" / "phase0_reproduction_task_sanity.yaml",
        output_dir,
        batch_size=3,
    )

    assert result == output_dir.resolve()
    assert (result / "trial_metadata.yaml").is_file()
    assert (result / "figures" / "task_trials.png").is_file()
