import logging
import numpy as np
import time
import copy
import random
from scipy.optimize import differential_evolution
from typing import Callable, List, Tuple, Dict, TypeVar
from core.optimizer import Optimizer, OptimizerFactory

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic Type

@OptimizerFactory.register("differential_evolution")
class DifferentialEvolutionOptimizer:
    """Імплементація алгоритму диференційної еволюції (DE) для оптимізації логістичних маршрутів."""

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        strategy: str = "best1bin",
        max_iterations: int = 1000,
        popsize: int = 15,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        timeout: float = 60.0,
        verbose: bool = False,
    ):
        """
        Ініціалізує алгоритм диференційної еволюції.
        
        Args:
            bounds: Межі пошуку [(min1, max1), (min2, max2), ...]
            strategy: Стратегія мутації ('best1bin', 'rand1bin', тощо)
            max_iterations: Максимальна кількість ітерацій
            popsize: Розмір популяції (зазвичай 10-20 особин на вимірність)
            mutation: Діапазон коефіцієнтів мутації (default: (0.5, 1.0))
            recombination: Ймовірність кросоверу (0-1)
            timeout: Часовий ліміт у секундах
            verbose: Включити детальний лог
        """
        self.bounds = bounds
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.timeout = timeout
        self.verbose = verbose

    def optimize(
        self,
        objective_function: Callable[[List[float]], float],
        is_maximizing: bool = False,
        fixed_points: List[int] = None,  # Індекси точок, які не можна змінювати
    ) -> Tuple[List[float], float, List[Tuple[int, float]]]:
        """
        Запускає оптимізацію методом диференційної еволюції.

        Args:
            objective_function: Функція оцінки вартості маршруту
            is_maximizing: True якщо максимізуємо, False якщо мінімізуємо
            fixed_points: Індекси, які мають залишатися незмінними
            
        Returns:
            Tuple[List[float], float, List[Tuple[int, float]]]: 
                Оптимізоване рішення, його вартість та історія вартості
        """
        start_time = time.time()
        cost_history = []

        def wrapper_function(x):
            """Обгортка для врахування максимізації та фіксованих точок."""
            if fixed_points:
                x = np.array(x)
                for idx in fixed_points:
                    if idx < len(self.bounds):
                        x[idx] = self.bounds[idx][0]  # Фіксуємо на мінімальній межі
            return -objective_function(x) if is_maximizing else objective_function(x)

        # Виконуємо оптимізацію
        result = differential_evolution(
            wrapper_function,
            bounds=self.bounds,
            strategy=self.strategy,
            maxiter=self.max_iterations,
            popsize=self.popsize,
            mutation=self.mutation,
            recombination=self.recombination,
            polish=True,  # Додаткова локальна оптимізація
            disp=self.verbose,
            callback=lambda xk, convergence: cost_history.append((len(cost_history), wrapper_function(xk))),
        )

        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Оптимізація завершена за {elapsed_time:.2f} секунд.")
            print(f"Знайдено найкращий стан з значенням {result.fun} після {result.nit} ітерацій.")

        return result.x.tolist(), result.fun, cost_history

if __name__ == "__main__":
    # Функція для мінімізації (Rosenbrock)
    def rosenbrock(x):
        return sum(100.0 * (x[i+1] - x[i]**2.0)**2 + (1 - x[i])**2 for i in range(len(x)-1))

    # Межі пошуку (2D-простір)
    bounds = [(-5, 5), (-5, 5)]

    # Створюємо об'єкт оптимізатора
    optimizer = DifferentialEvolutionOptimizer(bounds, verbose=True)

    best_solution, best_cost, cost_history = optimizer.optimize(rosenbrock)

    print(f"Найкраще рішення: {best_solution}")
    print(f"Найкраща вартість: {best_cost}")

    import matplotlib.pyplot as plt

    # Побудова графіка вартості
    iterations, costs = zip(*cost_history)
    plt.plot(iterations, costs)
    plt.xlabel("Ітерації")
    plt.ylabel("Вартість")
    plt.title("Конвергенція алгоритму DE")
    plt.show()


