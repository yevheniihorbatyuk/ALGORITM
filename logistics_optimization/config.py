"""
Конфігураційний файл для проекту оптимізації логістичної мережі.
"""
import os
from typing import Dict, Any, List, Tuple

# Шляхи до файлів та директорій
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Параметри логування
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "app.log")

# Параметри розподілених обчислень
RAY_ADDRESS = None  # None для локального кластера, або адреса Ray кластера
RAY_NUM_CPUS = None  # None для використання всіх доступних CPU
DASK_SCHEDULER_ADDRESS = None  # None для локального кластера, або адреса Dask планувальника
DASK_NUM_WORKERS = None  # None для використання всіх доступних CPU
DASK_MEMORY_LIMIT = "4GB"  # Ліміт пам'яті на одного воркера

# Параметри логістичної мережі
DEFAULT_NUM_NODES = 20
DEFAULT_NUM_VEHICLES = 8
DEFAULT_NUM_PACKAGES = 100


# Конфігурації алгоритмів за замовчуванням
DEFAULT_ALGORITHM_CONFIGS = {
    "hill_climbing": {
        "max_iterations": 1000,
        "max_no_improvement": 100,
        "random_restarts": 5,
        "timeout": 60.0,
        "verbose": False
    },
    "steepest_descent": {
        "max_iterations": 1000,
        "max_no_improvement": 50,
        "timeout": 60.0,
        "verbose": False
    },
    "random_search": {
        "max_iterations": 1000,
        "timeout": 60.0,
        "verbose": False
    },
    "simulated_annealing": {
        "initial_temp": 100.0,
        "cooling_rate": 0.95,
        "min_temp": 0.01,
        "max_iterations": 1000,
        "reset_temp_schedule": None,
        "timeout": 60.0,
        "verbose": False
    },
    "ant_colony": {
        "n_ants": 20,
        "n_iterations": 100,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.5,
        "q0": 0.9,
        "initial_pheromone": 0.1,
        "min_pheromone": 0.001,
        "max_pheromone": 1.0
    }
}

# Параметри для порівняння алгоритмів
COMPARISON_NUM_RUNS = 5  # Кількість запусків кожного алгоритму для усереднення результатів
BENCHMARK_TIMEOUT = 300.0  # Максимальний час виконання бенчмарку в секундах

# Параметри візуалізації
PLOT_FIGSIZE = (12, 8)  # Розмір фігур за замовчуванням
PLOT_DPI = 100  # Роздільна здатність для збереження графіків
PLOT_FORMAT = "png"  # Формат для збереження графіків (png, pdf, svg, ...)

# Режими демонстрації
DEMO_MODES = ["all", "comparison", "algorithms", "flow", "integration"]

# Функція для завантаження додаткових налаштувань з файлу
def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """
    Завантажує додаткові налаштування з файлу.
    
    Args:
        config_file: Шлях до файлу конфігурації
        
    Returns:
        Dict[str, Any]: Словник з налаштуваннями
    """
    if not os.path.exists(config_file):
        return {}
    
    config = {}
    with open(config_file, "r") as f:
        # Використовуємо exec для виконання Python коду з файлу
        # Це дозволяє використовувати складні структури даних і вирази в конфігурації
        exec(f.read(), globals(), config)
    
    return config

# Функція для отримання конфігурації алгоритму
def get_algorithm_config(algorithm_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Повертає конфігурацію для заданого алгоритму з можливістю перевизначення параметрів.
    
    Args:
        algorithm_name: Назва алгоритму
        overrides: Словник з параметрами, які треба перевизначити
        
    Returns:
        Dict[str, Any]: Конфігурація алгоритму
    """
    if algorithm_name not in DEFAULT_ALGORITHM_CONFIGS:
        raise ValueError(f"Алгоритм {algorithm_name} не знайдено в конфігурації")
    
    config = DEFAULT_ALGORITHM_CONFIGS[algorithm_name].copy()
    
    if overrides:
        config.update(overrides)
    
    return config

# Функція для забезпечення існування директорій
def ensure_directories():
    """Створює необхідні директорії, якщо вони не існують."""
    for directory in [DATA_DIR, RESULTS_DIR, os.path.dirname(LOG_FILE)]:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Створюємо необхідні директорії при імпорті
ensure_directories()