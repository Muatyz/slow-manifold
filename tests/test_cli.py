from pathlib import Path

import pytest

from slow_manifold.cli import build_parser


@pytest.mark.parametrize("command", ["task-sanity", "train", "visualize"])
def test_required_positional_command_argument(command: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args([command])

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("command", "experiment"),
    [
        ("task-sanity", "experiments/phase0_task_sanity.yaml"),
        ("train", "experiments/phase1_rank2_baseline.yaml"),
    ],
)
def test_explicit_experiment_argument_is_parsed(
    command: str, experiment: str
) -> None:
    args = build_parser().parse_args([command, "--experiment", experiment])

    assert args.experiment == Path(experiment)
    assert args.output_dir is None


def test_train_tag_is_parsed() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--experiment",
            "experiments/phase1_rank2_baseline.yaml",
            "--tag",
            "lr1e-4",
        ]
    )

    assert args.tag == "lr1e-4"


def test_visualize_run_argument_is_parsed() -> None:
    args = build_parser().parse_args(
        [
            "visualize",
            "--run",
            "runs/phase1_rank2_baseline/seed-20260903",
            "--arrow-stride",
            "1",
        ]
    )

    assert args.run == Path("runs/phase1_rank2_baseline/seed-20260903")
    assert args.arrow_stride == 1
    assert args.coordinate_bounds is None
