from __future__ import annotations

from ..config import DEFAULT_CONFIG
from ..data.synthetic import generate_problem


def test_generate_problem_shapes() -> None:
    problem = generate_problem(
        num_orders=DEFAULT_CONFIG.num_orders,
        num_vehicles=DEFAULT_CONFIG.num_vehicles,
        vehicle_capacity=DEFAULT_CONFIG.vehicle_capacity,
        depot_location=DEFAULT_CONFIG.depot_location,
        seed=DEFAULT_CONFIG.seed,
    )

    assert len(problem.orders) == DEFAULT_CONFIG.num_orders
    assert len(problem.vehicles) == DEFAULT_CONFIG.num_vehicles
    assert all(order.demand > 0 for order in problem.orders.values())


def test_generate_problem_deterministic() -> None:
    first = generate_problem(
        num_orders=5,
        num_vehicles=2,
        vehicle_capacity=100.0,
        depot_location=(0.0, 0.0),
        seed=123,
    )
    second = generate_problem(
        num_orders=5,
        num_vehicles=2,
        vehicle_capacity=100.0,
        depot_location=(0.0, 0.0),
        seed=123,
    )

    first_locations = sorted((order.location.x, order.location.y) for order in first.orders.values())
    second_locations = sorted((order.location.x, order.location.y) for order in second.orders.values())
    assert first_locations == second_locations

    first_demands = sorted(order.demand for order in first.orders.values())
    second_demands = sorted(order.demand for order in second.orders.values())
    assert first_demands == second_demands
