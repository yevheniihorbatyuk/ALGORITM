from __future__ import annotations

from ..algorithms.constructive import build_greedy_routes
from ..config import DEFAULT_CONFIG
from ..data.synthetic import generate_problem


def test_build_greedy_routes_produces_feasible_solution() -> None:
    problem = generate_problem(
        num_orders=DEFAULT_CONFIG.num_orders,
        num_vehicles=DEFAULT_CONFIG.num_vehicles,
        vehicle_capacity=DEFAULT_CONFIG.vehicle_capacity,
        depot_location=DEFAULT_CONFIG.depot_location,
        seed=DEFAULT_CONFIG.seed,
    )

    routes = build_greedy_routes(problem)

    assert problem.has_unique_assignments(routes)
    assert all(problem.is_capacity_feasible(route) for route in routes)
