"""Configuration helpers for the week 05 logistics optimisation scenario."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """High level knobs for running a logistics optimisation experiment.

    Attributes
    ----------
    num_orders:
        Number of delivery requests that should be generated.
    num_vehicles:
        How many vehicles should be available in the fleet.
    vehicle_capacity:
        Capacity (in arbitrary weight units) for every vehicle.
    depot_location:
        The (x, y) coordinates of the central depot.  Orders will be generated
        around this point.
    seed:
        Random seed used for data generation.  Keeping this value constant
        allows reproducible experiments.
    max_iterations:
        Number of local-search iterations that should be executed when improving
        the constructive solution.
    temperature:
        Starting temperature for the simulated annealing acceptance rule.  The
        value is intentionally exposed so that students can experiment with the
        cooling behaviour during week exercises.
    cooling_rate:
        Multiplicative factor used to cool down the temperature after each
        iteration.  Lower values result in quicker convergence at the expense of
        exploration.
    """

    num_orders: int = 20
    num_vehicles: int = 5
    vehicle_capacity: float = 120.0
    depot_location: tuple[float, float] = (0.0, 0.0)
    seed: int = 42
    max_iterations: int = 150
    temperature: float = 10.0
    cooling_rate: float = 0.95


DEFAULT_CONFIG = SimulationConfig()
"""Default parameters used by :mod:`projects.week_05.cli`."""
