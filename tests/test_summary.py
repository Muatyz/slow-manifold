from pathlib import Path

from slow_manifold.config import resolve_experiment
from slow_manifold.utils import startup_summary_lines

ROOT = Path(__file__).parents[1]


def test_startup_summary_reads_resolved_component_values() -> None:
    experiment = resolve_experiment(
        ROOT / "experiments" / "phase1_rank2_baseline.yaml"
    )

    lines = startup_summary_lines(
        experiment.components, experiment.components["reporting"]
    )

    assert lines == (
        "Task: Interval Categorization | threshold=5 | waveform=square",
        "Model: Vanilla RNN | N=64 | rank=2 | activation=tanh",
        "Train: loss=phase_normalized_mse | lambda pre-response=1 | lambda response=1 | "
        "optimizer=adam | learning rate=1e-4 | batch size=64 | "
        "updates=400000 | validation every=10 | device=cuda | "
        "initial state std=0.1 | neural noise std=0.001",
    )


def test_startup_summary_can_be_disabled() -> None:
    assert startup_summary_lines({}, {"enabled": False}) == ()
