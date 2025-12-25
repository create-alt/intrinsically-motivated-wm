import jax.numpy as jnp
import numpy as np

from dreamerv3 import dormant


def test_mean_abs_activation_and_ratio():
    x = jnp.array(
        [
            [[1.0, -1.0], [0.0, 2.0]],
            [[2.0, 2.0], [0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    mean_abs = dormant.mean_abs_activation(x, bdims=2)
    np.testing.assert_allclose(
        np.asarray(mean_abs), np.array([0.75, 1.25], dtype=np.float32), rtol=1e-6
    )
    ratio = dormant.dormant_ratio(mean_abs, tau=1.0)
    assert np.isclose(float(ratio), 0.5)


def test_aggregate_metrics():
    means = [
        jnp.array([1.0, 3.0], dtype=jnp.float32),
        jnp.array([2.0], dtype=jnp.float32),
    ]
    ratio = dormant.aggregate_metrics(means, tau=1.0)
    assert np.isclose(float(ratio), 2.0 / 3.0)
