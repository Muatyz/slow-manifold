"""Command-line entry point for public project workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from slow_manifold.sanity import create_task_sanity_run
from slow_manifold.workflows import (
    rerun_low_rank_visualization,
    resume_low_rank_training,
    run_low_rank_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sanity_parser = subparsers.add_parser(
        "task-sanity", help="generate and inspect a configured task batch"
    )
    sanity_parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
    )
    sanity_parser.add_argument("--split", default="train")
    sanity_parser.add_argument("--batch-size", type=int)
    sanity_parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the config-derived run directory",
    )

    train_parser = subparsers.add_parser(
        "train", help="train a low-rank CTRNN and generate configured diagnostics"
    )
    train_source = train_parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument(
        "--experiment",
        type=Path,
    )
    train_source.add_argument(
        "--resume",
        type=Path,
        help="continue an interrupted or failed run from its latest checkpoint",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the config-derived run directory",
    )
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--device", choices=("cpu", "cuda"))

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="re-render figures of an existing run without retraining",
    )
    visualize_parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="path of an existing run directory (the one written by 'train')",
    )
    visualize_parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the run's figures/ directory",
    )
    visualize_parser.add_argument(
        "--config-source",
        choices=("current", "original", "last"),
        default="current",
        help=(
            "post-training config source: current experiment recipe (default), "
            "original run config, or latest successful stage configs"
        ),
    )
    visualize_parser.add_argument(
        "--experiment",
        type=Path,
        help=(
            "recipe supplying current analysis/visualization settings; only "
            "valid with --config-source current"
        ),
    )
    visualize_parser.add_argument(
        "--coordinate-bounds",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="explicit latent coordinate range (recomputes the vector field)",
    )
    visualize_parser.add_argument(
        "--grid-points",
        type=int,
        help="grid resolution per axis (recomputes the vector field)",
    )
    visualize_parser.add_argument(
        "--square-coordinate-bounds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="expand the shorter latent-coordinate span to make a square plane",
    )
    visualize_parser.add_argument(
        "--arrow-stride",
        type=int,
        help="vector sampling density: draw every n-th grid point",
    )
    visualize_parser.add_argument(
        "--arrow-length-fraction",
        type=float,
        help="arrow length as a fraction of the latent coordinate span",
    )
    visualize_parser.add_argument(
        "--arrow-width",
        type=float,
        help="arrow shaft width in Matplotlib quiver axis-width units",
    )
    visualize_parser.add_argument(
        "--trajectory-line-width",
        type=float,
        help="task-trajectory main line width; cue overlays scale with it",
    )
    visualize_parser.add_argument(
        "--snapshot-dpi",
        type=int,
        help="PNG resolution; does not change diagnostics",
    )
    visualize_parser.add_argument(
        "--movie-dpi",
        type=int,
        help="MP4 frame resolution; independent of PNG resolution",
    )
    visualize_parser.add_argument(
        "--render-movies",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable MP4 encoding without changing static figures",
    )
    visualize_parser.add_argument(
        "--representative-epochs",
        type=int,
        nargs="+",
        help=(
            "manually replace automatic checkpoint selection for snapshots "
            "and output panels"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "task-sanity":
        run_dir = create_task_sanity_run(
            args.experiment,
            args.output_dir,
            split=args.split,
            batch_size=args.batch_size,
        )
        print(f"Task sanity run written to: {run_dir}")
        return 0
    if args.command == "train":
        if args.resume is not None:
            if any(
                value is not None
                for value in (
                    args.output_dir,
                    args.epochs,
                    args.batch_size,
                    args.device,
                )
            ):
                raise ValueError(
                    "--resume uses the frozen run config and cannot be combined "
                    "with output/config overrides"
                )
            run_dir = resume_low_rank_training(args.resume)
        else:
            run_dir = run_low_rank_training(
                args.experiment,
                args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            )
        print(f"Training run written to: {run_dir}")
        return 0
    if args.command == "visualize":
        figures_dir = rerun_low_rank_visualization(
            args.run,
            args.output_dir,
            coordinate_bounds=args.coordinate_bounds,
            grid_points=args.grid_points,
            square_coordinate_bounds=args.square_coordinate_bounds,
            arrow_stride=args.arrow_stride,
            arrow_length_fraction=args.arrow_length_fraction,
            arrow_width=args.arrow_width,
            trajectory_line_width=args.trajectory_line_width,
            snapshot_dpi=args.snapshot_dpi,
            movie_dpi=args.movie_dpi,
            render_movies=args.render_movies,
            representative_epochs=args.representative_epochs,
            config_source=args.config_source,
            experiment=args.experiment,
        )
        print(f"Figures written to: {figures_dir}")
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")
