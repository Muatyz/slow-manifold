"""Command-line entry point for public project workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from slow_manifold.sanity import create_task_sanity_run
from slow_manifold.workflows import rerun_rank2_visualization, run_rank2_training


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
        help="override runs/<experiment.name>/<tag>-seed-<seed>",
    )

    train_parser = subparsers.add_parser(
        "train", help="train a rank-2 CTRNN and generate configured diagnostics"
    )
    train_parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        help="override runs/<experiment.name>/<tag>-seed-<seed>",
    )
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--device", choices=("cpu", "cuda"))
    train_parser.add_argument(
        "--tag",
        help=(
            "run-directory tag separating otherwise identical recipes "
            "(e.g. 'lr1e-4'); see --help of 'visualize' for re-rendering"
        ),
    )

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
        "--arrow-stride",
        type=int,
        help="vector sampling density: draw every n-th grid point",
    )
    visualize_parser.add_argument(
        "--representative-epochs",
        type=int,
        nargs="+",
        help="snapshot epochs for vector-field PNGs and output panels",
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
        run_dir = run_rank2_training(
            args.experiment,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            tag=args.tag,
        )
        print(f"Training run written to: {run_dir}")
        return 0
    if args.command == "visualize":
        figures_dir = rerun_rank2_visualization(
            args.run,
            args.output_dir,
            coordinate_bounds=args.coordinate_bounds,
            grid_points=args.grid_points,
            arrow_stride=args.arrow_stride,
            representative_epochs=args.representative_epochs,
        )
        print(f"Figures written to: {figures_dir}")
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")
