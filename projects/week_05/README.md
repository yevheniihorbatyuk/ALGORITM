# Week 05 – Logistics Network Metaheuristics

This project provides a minimal yet extensible implementation of a logistics routing
scenario that can be used to experiment with local-search and metaheuristic
algorithms discussed during week 5 of the course.  The code purposefully mirrors
the rest of the repository structure so that future weeks can reuse the same
conventions.

## Contents

- `models/` – Core dataclasses describing the logistics entities and helper
  utilities for computing distances and validating capacity constraints.
- `data/` – Synthetic data generation helpers for quickly assembling reproducible
  problem instances.
- `algorithms/` – Baseline constructive heuristics and simple local-search
  improvements.
- `evaluation/` – Utilities for analysing produced routes.
- `cli.py` – A tiny command-line entry point for running a simulation from a
  configuration object.

## Quick start

```bash
python -m projects.week_05.cli
```

The default configuration generates a synthetic problem instance, constructs an
initial set of routes using the greedy heuristic, improves them with a
neighbourhood search, and finally prints an evaluation summary.  Every component
is written with the Python standard library so it can run inside simple execution
environments without external dependencies.
