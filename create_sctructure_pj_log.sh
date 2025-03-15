#!/bin/bash

# Створення кореневої директорії
mkdir -p logistics_optimization

# Перехід в кореневу директорію
cd logistics_optimization

# Створення директорії core та файлів
mkdir -p core
touch core/__init__.py core/model.py core/simulation.py core/optimizer.py

# Створення директорії algorithms та файлів
mkdir -p algorithms
touch algorithms/__init__.py

# Створення директорії local_search та файлів
mkdir -p algorithms/local_search
touch algorithms/local_search/__init__.py algorithms/local_search/hill_climbing.py algorithms/local_search/steepest_descent.py algorithms/local_search/random_search.py

# Створення файлу simulated_annealing.py
touch algorithms/simulated_annealing.py

# Створення директорії approximation та файлів
mkdir -p algorithms/approximation
touch algorithms/approximation/__init__.py algorithms/approximation/greedy.py

# Створення директорії randomized та файлів
mkdir -p algorithms/randomized
touch algorithms/randomized/__init__.py algorithms/randomized/monte_carlo.py

# Створення файлу markov_chains.py
touch algorithms/markov_chains.py

# Створення директорії swarm та файлів
mkdir -p algorithms/swarm
touch algorithms/swarm/__init__.py algorithms/swarm/particle_swarm.py algorithms/swarm/ant_colony.py

# Створення директорії execution та файлів
mkdir -p execution
touch execution/__init__.py execution/local_executor.py execution/ray_executor.py execution/dask_executor.py

# Створення директорії data та файлів
mkdir -p data/sample_data
touch data/__init__.py data/loaders.py data/generators.py

# Створення директорії analysis та файлів
mkdir -p analysis
touch analysis/__init__.py analysis/metrics.py analysis/visualization.py

# Створення директорії notebooks та файлів
mkdir -p notebooks
touch notebooks/algorithm_comparison.ipynb notebooks/scaling_analysis.ipynb notebooks/route_visualization.ipynb

# Створення директорії tests та файлів
mkdir -p tests
touch tests/__init__.py tests/test_model.py tests/test_algorithms.py tests/test_execution.py

# Створення файлів config.py, main.py та README.md
touch config.py main.py README.md

# Вихід з кореневої директорії
cd ..
