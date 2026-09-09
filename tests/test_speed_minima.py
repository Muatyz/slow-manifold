import numpy as np
import torch

from slow_manifold.analysis import (
    SpeedMinimumClassificationConfig,
    classify_speed_minimum,
)
from slow_manifold.analysis.speed_minima import refine_and_classify_speed_minima
from slow_manifold.models import Rank2CTRNN, Rank2CTRNNConfig


def test_dinc_style_classification_hierarchy() -> None:
    config = SpeedMinimumClassificationConfig()

    assert (
        classify_speed_minimum(
            speed=1.0e-7,
            eigenvalues=np.array([-0.2, -1.0]),
            is_local_minimum=True,
            config=config,
        )
        == "fixed_point"
    )
    assert (
        classify_speed_minimum(
            speed=1.0e-3,
            eigenvalues=np.array([1.0e-3, -0.4]),
            is_local_minimum=True,
            config=config,
        )
        == "latent_ghost_candidate"
    )
    assert (
        classify_speed_minimum(
            speed=1.0e-3,
            eigenvalues=np.array([1.0e-3, 0.4]),
            is_local_minimum=True,
            config=config,
        )
        == "slow_point"
    )
    assert (
        classify_speed_minimum(
            speed=1.0e-3,
            eigenvalues=np.array([0.0, -0.4]),
            is_local_minimum=False,
            config=config,
        )
        == "unresolved"
    )


def test_refinement_finds_linear_fixed_attractor() -> None:
    model_config = Rank2CTRNNConfig(
        name="linear_test",
        input_size=1,
        state_size=4,
        rank=2,
        output_size=1,
        tau=1.0,
        dt=0.1,
        dtype="float32",
        factor_scale=0.0,
        input_scale=0.0,
        bias_scale=0.0,
        readout_scale=0.0,
    )
    model = Rank2CTRNN(model_config)
    model.requires_grad_(False)
    basis = torch.eye(4, 2)
    values = np.linspace(-1.0, 1.0, 5)
    grid_x, grid_y = np.meshgrid(values, values)
    mask = np.zeros((5, 5), dtype=np.bool_)
    mask[2, 2] = True

    result = refine_and_classify_speed_minima(
        model=model,
        basis=basis,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_minimum_mask=mask,
        input_condition=(0.0,),
        bounds=(-1.0, 1.0, -1.0, 1.0),
        config=SpeedMinimumClassificationConfig(),
    )

    np.testing.assert_allclose(result.coordinates, [[0.0, 0.0]], atol=1.0e-7)
    assert result.classification.tolist() == ["fixed_point"]
    assert result.is_attractor.tolist() == [True]
    assert result.is_local_minimum.tolist() == [True]
