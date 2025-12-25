from __future__ import annotations

from typing import Any

import elements


def dormant_enabled(agent) -> bool:
    """Return whether dormant metrics are enabled for the agent.

    Args:
        agent: Agent instance that may expose a dormant config.

    Returns:
        bool: True if dormant metrics are enabled, otherwise False.
    """
    try:
        return bool(agent.config.dormant.enable)
    except AttributeError:
        return False


def split_dormant_metrics(
    metrics: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split dormant metrics from the rest of the report metrics.

    Args:
        metrics (dict[str, Any] | None): Metric mapping to split.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Dormant metrics and non-dormant
            metrics.
    """
    if not metrics:
        return {}, {}
    dormant = {}
    other = {}
    for key, value in metrics.items():
        if key.startswith("dormant/"):
            dormant[key] = value
        else:
            other[key] = value
    return dormant, other


def collect_dormant_metrics(agent, carry, stream, batches: int):
    """Aggregate dormant metrics from agent.report.

    Args:
        agent: Agent instance exposing report().
        carry: Report carry state.
        stream: Iterator yielding report batches.
        batches (int): Number of batches to aggregate.

    Returns:
        tuple: Updated carry and dormant metrics dict.
    """
    batches = int(batches)
    if batches <= 0:
        return carry, {}
    agg = elements.Agg()
    has_metrics = False
    for _ in range(batches):
        carry, mets = agent.report(carry, next(stream))
        dormant_metrics, _ = split_dormant_metrics(mets)
        if dormant_metrics:
            agg.add(dormant_metrics)
            has_metrics = True
    if not has_metrics:
        return carry, {}
    return carry, agg.result()
