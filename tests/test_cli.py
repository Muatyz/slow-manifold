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


def test_train_rejects_removed_tag_argument() -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            [
                "train",
                "--experiment",
                "experiments/phase1_rank2_baseline.yaml",
                "--tag",
                "lr1e-4",
            ]
        )

    assert error.value.code == 2


def test_train_resume_argument_is_parsed() -> None:
    args = build_parser().parse_args(
        ["train", "--resume", "runs/interrupted/seed-17"]
    )

    assert args.resume == Path("runs/interrupted/seed-17")
    assert args.experiment is None


def test_visualize_run_argument_is_parsed() -> None:
    args = build_parser().parse_args(
        [
            "visualize",
            "--run",
            "runs/phase1_rank2_baseline/seed-20260903",
            "--arrow-stride",
            "1",
            "--arrow-length-fraction",
            "0.0075",
            "--arrow-width",
            "0.0012",
            "--trajectory-line-width",
            "0.8",
            "--snapshot-dpi",
            "300",
            "--movie-dpi",
            "120",
            "--no-render-movies",
            "--config-source",
            "last",
        ]
    )

    assert args.run == Path("runs/phase1_rank2_baseline/seed-20260903")
    assert args.arrow_stride == 1
    assert args.arrow_length_fraction == pytest.approx(0.0075)
    assert args.arrow_width == pytest.approx(0.0012)
    assert args.trajectory_line_width == pytest.approx(0.8)
    assert args.snapshot_dpi == 300
    assert args.movie_dpi == 120
    assert args.render_movies is False
    assert args.coordinate_bounds is None
    assert args.square_coordinate_bounds is None
    assert args.config_source == "last"
    assert args.experiment is None


def test_visualize_current_experiment_argument_is_parsed() -> None:
    args = build_parser().parse_args(
        [
            "visualize",
            "--run",
            "runs/example",
            "--experiment",
            "experiments/phase1_rank2_baseline.yaml",
        ]
    )

    assert args.config_source == "current"
    assert args.experiment == Path("experiments/phase1_rank2_baseline.yaml")


def test_visualize_square_coordinate_override_is_parsed() -> None:
    args = build_parser().parse_args(
        ["visualize", "--run", "runs/example", "--no-square-coordinate-bounds"]
    )

    assert args.square_coordinate_bounds is False
