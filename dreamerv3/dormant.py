"""Dormant neuron metrics for DreamerV3."""

from __future__ import annotations

import math
from typing import Iterable

import jax.numpy as jnp


def mean_abs_activation(x: jnp.ndarray | None, bdims: int) -> jnp.ndarray | None:
    """Compute per-feature mean absolute activation.

    Args:
        x (jnp.ndarray | None): Activation tensor.
        bdims (int): Number of leading batch dimensions treated as samples.

    Returns:
        jnp.ndarray | None: Mean absolute activation per feature, or None if invalid.
    """
    if x is None:
        return None
    if x.ndim <= bdims:
        return None
    sample_shape = x.shape[:bdims]
    if 0 in sample_shape:
        return None
    sample_count = math.prod(sample_shape) if sample_shape else 1
    flat = x.reshape((sample_count, -1))
    return jnp.mean(jnp.abs(flat), axis=0, dtype=jnp.float32)


def dormant_ratio(mean_abs: jnp.ndarray | None, tau: float) -> jnp.ndarray | None:
    """Compute dormant ratio from per-feature mean absolute activations.

    Args:
        mean_abs (jnp.ndarray | None): Per-feature mean absolute activation.
        tau (float): Dormant threshold.

    Returns:
        jnp.ndarray | None: Dormant ratio, or None if input is invalid.
    """
    if mean_abs is None:
        return None
    if mean_abs.size == 0:
        return None
    layer_mean = jnp.mean(mean_abs, dtype=jnp.float32)
    scores = mean_abs / (layer_mean + 1e-8)
    dormant = scores <= tau
    return jnp.mean(dormant, dtype=jnp.float32)


def aggregate_metrics(
    mean_abs_list: Iterable[jnp.ndarray | None], tau: float
) -> jnp.ndarray | None:
    """Aggregate dormant ratio over concatenated feature means.

    Args:
        mean_abs_list (Iterable[jnp.ndarray | None]): Per-layer mean abs arrays.
        tau (float): Dormant threshold.

    Returns:
        jnp.ndarray | None: Dormant ratio for concatenated features, or None.
    """
    means = [mean_abs for mean_abs in mean_abs_list if mean_abs is not None]
    means = [mean_abs for mean_abs in means if mean_abs.size]
    if not means:
        return None
    stacked = jnp.concatenate(means, axis=0)
    return dormant_ratio(stacked, tau)
