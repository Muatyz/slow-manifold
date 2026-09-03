from pathlib import Path

import pytest

from slow_manifold.sanity import create_task_sanity_run

ROOT = Path(__file__).parents[1]


def test_task_sanity_workflow_writes_traceable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "runs" / "phase0_task_sanity" / "seed-20260903"

    result = create_task_sanity_run(
        ROOT / "experiments" / "phase0_task_sanity.yaml",
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
