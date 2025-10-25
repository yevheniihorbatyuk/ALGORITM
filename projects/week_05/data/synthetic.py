"""Synthetic data helpers for the week 05 project."""
from __future__ import annotations

from typing import Iterable
import random

from ..models.logistics import Location, LogisticsProblem, Order, Vehicle


def generate_locations(
    n: int,
    *,
    center: tuple[float, float],
    scale: float,
    rng: random.Random,
) -> Iterable[Location]:
    """Generate ``n`` random :class:`Location` instances around ``center``."""

    cx, cy = center
    for idx in range(n):
        yield Location(
            id=f"loc-{idx}",
            x=cx + rng.gauss(0.0, scale),
            y=cy + rng.gauss(0.0, scale),
        )


def generate_problem(
    *,
    num_orders: int,
    num_vehicles: int,
    vehicle_capacity: float,
    depot_location: tuple[float, float],
    seed: int,
) -> LogisticsProblem:
    """Generate a reproducible :class:`LogisticsProblem` instance."""

    if num_orders <= 0:
        raise ValueError("num_orders must be positive")
    if num_vehicles <= 0:
        raise ValueError("num_vehicles must be positive")
    if vehicle_capacity <= 0:
        raise ValueError("vehicle_capacity must be positive")

    rng = random.Random(seed)

    depot = Location(id="depot", x=float(depot_location[0]), y=float(depot_location[1]))
    locations = list(generate_locations(num_orders, center=depot_location, scale=5.0, rng=rng))

    orders = [
        Order(id=f"order-{idx}", location=location, demand=rng.uniform(5.0, 25.0))
        for idx, location in enumerate(locations)
    ]

    vehicles = [Vehicle(id=f"vehicle-{idx}", capacity=vehicle_capacity) for idx in range(num_vehicles)]

    return LogisticsProblem(depot=depot, orders=orders, vehicles=vehicles)


__all__ = ["generate_problem", "generate_locations"]
