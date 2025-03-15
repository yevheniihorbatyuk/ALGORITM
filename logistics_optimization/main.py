# Імпорт бібліотек та модулів
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import copy
from typing import List, Dict, Tuple, Any

# Встановлюємо стиль графіків
plt.style.use('ggplot')

# Імпортуємо основні модулі
from core.model import LogisticsNetwork, NodeType, VehicleType, PackagePriority
from core.optimizer import OptimizerFactory
from execution.ray_executor import RayExecutor
from execution.dask_executor import DaskExecutor
from analysis.metrics import LogisticsMetrics

from algorithms.local_search.hill_climbing import HillClimbing, SteepestDescent, RandomSearch
from algorithms.simulated_annealing import SimulatedAnnealing
# from algorithms.swarm.ant_colony import AntColonyOptimization



def create_sample_network(num_nodes: int = 15, num_vehicles: int = 5, num_packages: int = 50) -> LogisticsNetwork:
    """Створює зразкову логістичну мережу для тестування."""
    # Створюємо мережу
    network = LogisticsNetwork()
    
    # Додаємо випадкові дані
    network.create_random_network(num_nodes, num_vehicles, num_packages)
    
    return network

# Створюємо мережу
network = create_sample_network(20, 8, 100)

# Виводимо інформацію про мережу
print(f"Створено мережу з {len(network.nodes)} вузлами, {len(network.vehicles)} транспортними засобами та {len(network.packages)} посилками")

# Створюємо об'єкт для обчислення метрик
metrics = LogisticsMetrics(network)

# Обчислюємо загальні метрики мережі
network_metrics = metrics.calculate_network_metrics()

# Виводимо деякі цікаві метрики
print(f"\nРозподіл вузлів за типами:")
for node_type, count in network_metrics["node_counts"].items():
    print(f"  {node_type}: {count}")

print(f"\nРозподіл транспортних засобів за типами:")
for vehicle_type, count in network_metrics["vehicle_counts"].items():
    print(f"  {vehicle_type}: {count}")

print(f"\nРозподіл посилок за пріоритетами:")
for priority, count in network_metrics["package_counts"].items():
    print(f"  {priority}: {count}")

print(f"\nЗв'язність мережі: {network_metrics['connectivity_ratio']:.2f}")
print(f"Середня вага посилки: {network_metrics['avg_package_weight']:.2f} кг")
print(f"Середній об'єм посилки: {network_metrics['avg_package_volume']:.4f} м³")