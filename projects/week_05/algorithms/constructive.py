"""Baseline constructive heuristics used as starting point for local-search."""
from __future__ import annotations

from typing import Dict, Iterable, List

from ..models.logistics import LogisticsProblem, Order, Route


def _sorted_orders(problem: LogisticsProblem) -> List[Order]:
    orders = list(problem.orders.values())
    orders.sort(key=lambda order: problem.distance("__depot__", order.id))
    return orders


def build_greedy_routes(problem: LogisticsProblem) -> List[Route]:
    """Assign orders to vehicles with a simple greedy heuristic.

    Orders are processed in the order of their distance from the depot.  Each
    order is assigned to the route that currently has the highest remaining
    capacity.  The heuristic is intentionally simple so that it can be improved
    upon with local-search in :mod:`projects.week_05.algorithms.local_search`.
    """

    routes = [Route(vehicle_id=vehicle_id) for vehicle_id in problem.vehicles]
    remaining_capacity: Dict[str, float] = {
        vehicle_id: problem.vehicles[vehicle_id].capacity for vehicle_id in problem.vehicles
    }

    for order in _sorted_orders(problem):
        feasible_route = _select_route(order.demand, routes, remaining_capacity)
        if feasible_route is None:
            raise ValueError(
                "Unable to assign order to any vehicle.  Increase vehicle capacity or fleet size."
            )
        feasible_route.order_ids.append(order.id)
        remaining_capacity[feasible_route.vehicle_id] -= order.demand

    return routes


def _select_route(demand: float, routes: Iterable[Route], remaining_capacity: Dict[str, float]) -> Route | None:
    best_route: Route | None = None
    best_remaining = float("-inf")
    for route in routes:
        capacity_left = remaining_capacity[route.vehicle_id]
        if capacity_left >= demand and capacity_left > best_remaining:
            best_remaining = capacity_left
            best_route = route
    return best_route


__all__ = ["build_greedy_routes"]
