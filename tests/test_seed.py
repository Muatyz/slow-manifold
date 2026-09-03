import numpy as np

from slow_manifold.utils import make_rng


def test_named_rng_streams_are_reproducible_and_independent() -> None:
    train_a = make_rng(123, "task:train").random(8)
    train_b = make_rng(123, "task:train").random(8)
    validation = make_rng(123, "task:validation").random(8)

    np.testing.assert_array_equal(train_a, train_b)
    assert not np.array_equal(train_a, validation)
