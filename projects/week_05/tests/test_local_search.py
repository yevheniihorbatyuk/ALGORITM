from __future__ import annotations

from ..algorithms.constructive import build_greedy_routes
from ..algorithms.local_search import optimise_routes
from ..config import DEFAULT_CONFIG
from ..data.synthetic import generate_problem
from ..evaluation.metrics import total_distance


def test_optimise_routes_not_worse_than_greedy() -> None:
    config = DEFAULT_CONFIG
    problem = generate_problem(
        num_orders=config.num_orders,
        num_vehicles=config.num_vehicles,
        vehicle_capacity=config.vehicle_capacity,
        depot_location=config.depot_location,
        seed=config.seed,
    )

    greedy = build_greedy_routes(problem)
    improved = optimise_routes(
        problem,
        greedy,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
        cooling_rate=config.cooling_rate,
        seed=config.seed,
    )

    assert total_distance(problem, improved) <= total_distance(problem, greedy) + 1e-9
