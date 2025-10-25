"""Local-search utilities for improving constructive solutions."""
from __future__ import annotations

import math
import random
from typing import List, Sequence

from ..evaluation.metrics import total_distance
from ..models.logistics import LogisticsProblem, Route


def optimise_routes(
    problem: LogisticsProblem,
    routes: Sequence[Route],
    *,
    max_iterations: int,
    temperature: float,
    cooling_rate: float,
    seed: int,
) -> List[Route]:
    """Improve ``routes`` using a lightweight simulated annealing procedure."""

    rng = random.Random(seed)

    current = [route.clone() for route in routes]
    current_cost = total_distance(problem, current)
    best = [route.clone() for route in current]
    best_cost = current_cost

    candidate_indices = [idx for idx, route in enumerate(current) if len(route.order_ids) > 1]
    if not candidate_indices:
        return best

    temp = float(temperature)
    for _ in range(max_iterations):
        idx = rng.choice(candidate_indices)
        candidate_route = current[idx]
        swap_i, swap_j = _pick_swap_indices(candidate_route.order_ids, rng)

        new_routes = [route.clone() for route in current]
        new_route = new_routes[idx]
        new_route.order_ids[swap_i], new_route.order_ids[swap_j] = (
            new_route.order_ids[swap_j],
            new_route.order_ids[swap_i],
        )

        new_cost = total_distance(problem, new_routes)
        delta = new_cost - current_cost

        if delta < 0 or _accept_move(delta, temp, rng.random()):
            current = new_routes
            current_cost = new_cost
            candidate_indices = [idx for idx, route in enumerate(current) if len(route.order_ids) > 1]
            if current_cost < best_cost:
                best = [route.clone() for route in current]
                best_cost = current_cost

        temp *= cooling_rate
        if temp <= 1e-6:
            break

    return best


def _pick_swap_indices(order_ids: Sequence[str], rng: random.Random) -> tuple[int, int]:
    first = rng.randrange(len(order_ids))
    second = rng.randrange(len(order_ids) - 1)
    if second >= first:
        second += 1
    return min(first, second), max(first, second)


def _accept_move(delta: float, temperature: float, probability: float) -> bool:
    if delta < 0:
        return True
    if temperature <= 0:
        return False
    threshold = math.exp(-delta / temperature)
    return probability < threshold


__all__ = ["optimise_routes"]
