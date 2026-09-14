from pathlib import Path

import torch

from slow_manifold.config import resolve_experiment
from slow_manifold.models import Rank2CTRNN, Rank2CTRNNConfig

ROOT = Path(__file__).parents[1]


def make_model() -> Rank2CTRNN:
    experiment = resolve_experiment(ROOT / "experiments" / "phase1_rank2_baseline.yaml")
    config = Rank2CTRNNConfig.from_mapping(experiment.components["model"])
    generator = torch.Generator().manual_seed(12)
    return Rank2CTRNN(config, generator=generator)


def test_recurrent_matrix_has_rank_at_most_two() -> None:
    model = make_model()
    assert torch.linalg.matrix_rank(model.recurrent_matrix).item() <= 2


def test_full_and_exact_latent_flows_agree() -> None:
    model = make_model()
    state = torch.randn(5, model.config.state_size)
    inputs = torch.randn(5, model.config.input_size)

    projected_full_flow = model.flow(state, inputs) @ model.n
    exact_latent_flow = model.latent_flow(model.latent(state), inputs)

    torch.testing.assert_close(projected_full_flow, exact_latent_flow)


def test_row_space_jacobian_matches_autograd() -> None:
    model = make_model()
    basis = torch.linalg.qr(model.n.detach(), mode="reduced").Q
    coordinates = torch.randn(3, model.config.rank)
    inputs = torch.randn(3, model.config.input_size)

    analytic = model.row_space_jacobian(coordinates, inputs, basis)
    automatic = torch.stack(
        [
            torch.autograd.functional.jacobian(
                lambda point: model.row_space_flow(
                    point.unsqueeze(0), inputs[index : index + 1], basis
                ).squeeze(0),
                coordinates[index],
            )
            for index in range(coordinates.shape[0])
        ]
    )

    torch.testing.assert_close(analytic, automatic, rtol=1e-5, atol=1e-6)


def test_rollout_shapes_are_batch_first() -> None:
    model = make_model()
    inputs = torch.zeros(4, 11, model.config.input_size)
    outputs, states, latents = model.rollout(inputs)

    assert outputs.shape == (4, 11, 1)
    assert states.shape == (4, 11, model.config.state_size)
    assert latents.shape == (4, 11, 2)


def test_tanh_normal_initial_state_is_seeded() -> None:
    model = make_model()
    actual_generator = torch.Generator().manual_seed(41)
    expected_generator = torch.Generator().manual_seed(41)

    actual = model.sample_initial_state(
        3, mean=0.0, std=0.1, generator=actual_generator
    )
    expected = torch.tanh(
        torch.empty(3, model.config.state_size).normal_(
            0.0, 0.1, generator=expected_generator
        )
    )

    torch.testing.assert_close(actual, expected)


def test_neural_noise_is_added_inside_tanh_before_euler_step() -> None:
    model = make_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    inputs = torch.zeros(2, 1, model.config.input_size)
    actual_generator = torch.Generator().manual_seed(73)
    expected_generator = torch.Generator().manual_seed(73)
    noise_std = 0.2

    _, states, _ = model.rollout(
        inputs,
        neural_noise_std=noise_std,
        generator=actual_generator,
    )
    noise = torch.empty(2, model.config.state_size).normal_(
        0.0, noise_std, generator=expected_generator
    )
    expected = model.config.dt / model.config.tau * torch.tanh(noise)

    torch.testing.assert_close(states[:, 0], expected)


def test_zero_noise_arguments_preserve_deterministic_rollout() -> None:
    model = make_model()
    inputs = torch.zeros(2, 5, model.config.input_size)

    baseline = model.rollout(inputs)
    explicit_zero = model.rollout(
        inputs,
        neural_noise_mean=0.0,
        neural_noise_std=0.0,
        generator=torch.Generator().manual_seed(99),
    )

    for expected, actual in zip(baseline, explicit_zero):
        torch.testing.assert_close(actual, expected)
