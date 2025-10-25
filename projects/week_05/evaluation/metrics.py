"""Evaluation helpers for week 05 optimisation experiments."""
from __future__ import annotations

from typing import Iterable, Sequence

from ..models.logistics import LogisticsProblem, Route


def route_distance(problem: LogisticsProblem, route: Route) -> float:
    """Compute the total travelled distance for ``route`` including depot legs."""

    if not route.order_ids:
        return 0.0

    total = 0.0
    previous = "__depot__"
    for order_id in route.order_ids:
        total += problem.distance(previous, order_id)
        previous = order_id
    total += problem.distance(previous, "__depot__")
    return total


def total_distance(problem: LogisticsProblem, routes: Iterable[Route]) -> float:
    """Compute the total distance travelled by all vehicles."""

    return sum(route_distance(problem, route) for route in routes)


def average_capacity_utilisation(problem: LogisticsProblem, routes: Sequence[Route]) -> float:
    """Return the mean load factor across all routes.

    Empty routes are ignored when computing the average as they would otherwise
    skew the metric towards zero when the fleet size is intentionally larger
    than the number of required tours.
    """

    utilisations = []
    for route in routes:
        if not route.order_ids:
            continue
        vehicle = problem.get_vehicle(route.vehicle_id)
        utilisation = route.total_demand(problem.orders) / vehicle.capacity
        utilisations.append(utilisation)
    if not utilisations:
        return 0.0
    return sum(utilisations) / len(utilisations)


__all__ = ["route_distance", "total_distance", "average_capacity_utilisation"]
