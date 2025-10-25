"""Core data structures for the week 05 logistics optimisation exercises."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence
import math


@dataclass(frozen=True)
class Location:
    """Simple two dimensional coordinate."""

    id: str
    x: float
    y: float

    def distance_to(self, other: "Location") -> float:
        """Return the Euclidean distance to another location."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Order:
    """Delivery request that has to be fulfilled by a vehicle."""

    id: str
    location: Location
    demand: float


@dataclass(frozen=True)
class Vehicle:
    """Vehicle participating in the simulation."""

    id: str
    capacity: float


@dataclass
class Route:
    """A sequence of orders assigned to a single vehicle."""

    vehicle_id: str
    order_ids: List[str] = field(default_factory=list)

    def total_demand(self, orders: Mapping[str, Order]) -> float:
        """Return the total demand transported on this route."""
        return sum(orders[order_id].demand for order_id in self.order_ids)

    def clone(self) -> "Route":
        """Return a shallow copy of the route.

        Cloning is handy for metaheuristics that explore alternative
        neighbourhoods without mutating the original solution.
        """

        return Route(vehicle_id=self.vehicle_id, order_ids=list(self.order_ids))


class LogisticsProblem:
    """Container describing a single vehicle routing scenario."""

    def __init__(self, depot: Location, orders: Sequence[Order], vehicles: Sequence[Vehicle]):
        if not orders:
            raise ValueError("The problem requires at least one order to be useful.")
        if not vehicles:
            raise ValueError("At least one vehicle must be available for routing.")

        self.depot = depot
        self.orders: Dict[str, Order] = {order.id: order for order in orders}
        self.vehicles: Dict[str, Vehicle] = {vehicle.id: vehicle for vehicle in vehicles}
        self._distance_cache: Dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Helper accessors
    # ------------------------------------------------------------------
    def order_locations(self) -> Iterable[Location]:
        """Iterate through locations associated with all orders."""

        return (order.location for order in self.orders.values())

    def get_vehicle(self, vehicle_id: str) -> Vehicle:
        try:
            return self.vehicles[vehicle_id]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Unknown vehicle id: {vehicle_id}") from exc

    # ------------------------------------------------------------------
    # Distance utilities
    # ------------------------------------------------------------------
    def _lookup_location(self, identifier: str) -> Location:
        if identifier == "__depot__":
            return self.depot
        return self.orders[identifier].location

    def distance(self, first: str, second: str) -> float:
        """Compute the distance between the two identifiers.

        The special identifier ``"__depot__"`` is used internally for the depot
        location.  Order identifiers are resolved against the known orders.
        """

        cache_key = (first, second)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        loc_first = self._lookup_location(first)
        loc_second = self._lookup_location(second)
        value = loc_first.distance_to(loc_second)
        self._distance_cache[cache_key] = value
        self._distance_cache[(second, first)] = value
        return value

    def distance_matrix(self) -> List[List[float]]:
        """Return the full distance matrix between depot and all orders."""

        ids = ["__depot__"] + sorted(self.orders)
        size = len(ids)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        for i, first in enumerate(ids):
            for j in range(i + 1, size):
                second = ids[j]
                value = self.distance(first, second)
                matrix[i][j] = value
                matrix[j][i] = value
        return matrix

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def is_capacity_feasible(self, route: Route) -> bool:
        """Check whether a route respects the associated vehicle capacity."""

        vehicle = self.get_vehicle(route.vehicle_id)
        total = route.total_demand(self.orders)
        return total <= vehicle.capacity + 1e-9

    def has_unique_assignments(self, routes: Sequence[Route]) -> bool:
        """Ensure each order is assigned to at most one route."""

        seen: set[str] = set()
        for route in routes:
            for order_id in route.order_ids:
                if order_id in seen:
                    return False
                seen.add(order_id)
        return seen == set(self.orders)


__all__ = [
    "Location",
    "Order",
    "Vehicle",
    "Route",
    "LogisticsProblem",
]
