"""Simple command line entry point for the week 05 project."""
from __future__ import annotations

from dataclasses import asdict

from .algorithms.constructive import build_greedy_routes
from .algorithms.local_search import optimise_routes
from .config import DEFAULT_CONFIG, SimulationConfig
from .data.synthetic import generate_problem
from .evaluation.metrics import average_capacity_utilisation, total_distance


def run_simulation(config: SimulationConfig) -> None:
    """Execute the full optimisation workflow and print a short report."""

    problem = generate_problem(
        num_orders=config.num_orders,
        num_vehicles=config.num_vehicles,
        vehicle_capacity=config.vehicle_capacity,
        depot_location=config.depot_location,
        seed=config.seed,
    )

    initial_routes = build_greedy_routes(problem)
    improved_routes = optimise_routes(
        problem,
        initial_routes,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
        cooling_rate=config.cooling_rate,
        seed=config.seed,
    )

    initial_cost = total_distance(problem, initial_routes)
    improved_cost = total_distance(problem, improved_routes)
    utilisation = average_capacity_utilisation(problem, improved_routes)

    print("Configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    print("\nResults:")
    print(f"  Initial distance:  {initial_cost:.2f}")
    print(f"  Improved distance: {improved_cost:.2f}")
    print(f"  Utilisation:       {utilisation:.2%}")


def main() -> None:
    run_simulation(DEFAULT_CONFIG)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    main()
