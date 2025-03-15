from typing import List, Dict, Tuple, Callable, Any, TypeVar, Generic
import random
import copy
import time
import logging
from core.optimizer import Optimizer, OptimizerFactory

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

T = TypeVar('T')  # Тип стану (наприклад, маршрут або розподіл посилок)

@OptimizerFactory.register("hill_climbing")
class HillClimbing(Optimizer[T, Tuple[T, float, List[Tuple[int, float]]]]):
    """Реалізація алгоритму Hill Climbing для оптимізації логістичних задач."""
    
    def __init__(
        self,
        max_iterations: int = 1000,
        max_no_improvement: int = 100,
        timeout: float = 60.0,  # Максимальний час виконання у секундах
        random_restarts: int = 5,  # Кількість випадкових перезапусків
        verbose: bool = False
    ):
        self.max_iterations = max_iterations
        self.max_no_improvement = max_no_improvement
        self.timeout = timeout
        self.random_restarts = random_restarts
        self.verbose = verbose
    
    def optimize(
        self,
        initial_state: T,
        objective_function: Callable[[T], float],
        neighborhood_function: Callable[[T], List[T]],
        is_maximizing: bool = False,
        restart_function: Callable[[], T] = None,
        **kwargs
    ) -> Tuple[T, float, List[Tuple[int, float]]]:
        """
        Оптимізує стан за допомогою алгоритму Hill Climbing.
        
        Args:
            initial_state: Початковий стан
            objective_function: Функція оцінки стану
            neighborhood_function: Функція, що генерує сусідні стани
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
            restart_function: Функція для генерації нового початкового стану при перезапуску
            **kwargs: Додаткові параметри
            
        Returns:
            Tuple[T, float, List[Tuple[int, float]]]: 
                Оптимізований стан, його оцінка та історія оцінок
        """
        if restart_function is None:
            # Якщо функція перезапуску не надана, використовуємо копію початкового стану
            restart_function = lambda: copy.deepcopy(initial_state)
        
        best_state = copy.deepcopy(initial_state)
        best_value = objective_function(best_state)
        
        if self.verbose:
            print(f"Початковий стан має значення функції: {best_value}")
        
        history = [(0, best_value)]
        total_iterations = 0
        
        start_time = time.time()
        
        for restart in range(self.random_restarts):
            if restart > 0:
                # Починаємо з нового стану для кожного перезапуску (крім першого)
                current_state = restart_function()
                current_value = objective_function(current_state)
                if self.verbose:
                    print(f"Перезапуск {restart}: новий початковий стан має значення функції: {current_value}")
            else:
                current_state = copy.deepcopy(initial_state)
                current_value = best_value
            
            no_improvement = 0
            
            for iteration in range(self.max_iterations):
                total_iterations += 1
                
                # Перевіряємо таймаут
                if time.time() - start_time > self.timeout:
                    if self.verbose:
                        print(f"Досягнуто ліміт часу ({self.timeout} секунд). Зупиняємо оптимізацію.")
                    break
                
                # Генеруємо сусідні стани
                neighbors = neighborhood_function(current_state)
                
                # Якщо немає сусідів, зупиняємося
                if not neighbors:
                    if self.verbose:
                        print("Немає сусідніх станів. Зупиняємо оптимізацію.")
                    break
                
                # Оцінюємо сусідні стани
                neighbor_values = [objective_function(neighbor) for neighbor in neighbors]
                
                # Знаходимо найкращого сусіда
                if is_maximizing:
                    best_neighbor_idx = max(range(len(neighbor_values)), key=lambda i: neighbor_values[i])
                    best_neighbor_value = neighbor_values[best_neighbor_idx]
                    improvement = best_neighbor_value > current_value
                else:
                    best_neighbor_idx = min(range(len(neighbor_values)), key=lambda i: neighbor_values[i])
                    best_neighbor_value = neighbor_values[best_neighbor_idx]
                    improvement = best_neighbor_value < current_value
                
                # Якщо знайдено покращення, оновлюємо поточний стан
                if improvement:
                    current_state = neighbors[best_neighbor_idx]
                    current_value = best_neighbor_value
                    no_improvement = 0
                    
                    # Оновлюємо найкращий стан, якщо це необхідно
                    if (is_maximizing and current_value > best_value) or \
                       (not is_maximizing and current_value < best_value):
                        best_state = copy.deepcopy(current_state)
                        best_value = current_value
                        if self.verbose:
                            print(f"Ітерація {total_iterations}: знайдено новий найкращий стан з значенням {best_value}")
                else:
                    no_improvement += 1
                
                # Зберігаємо історію оцінок
                if total_iterations % 10 == 0:
                    history.append((total_iterations, current_value))
                
                # Зупиняємося, якщо немає покращення протягом певної кількості ітерацій
                if no_improvement >= self.max_no_improvement:
                    if self.verbose:
                        print(f"Немає покращення протягом {self.max_no_improvement} ітерацій. Перезапуск.")
                    break
            
            # Зберігаємо останню точку історії для цього перезапуску
            if total_iterations % 10 != 0:
                history.append((total_iterations, current_value))
            
            # Зупиняємося, якщо вичерпано час
            if time.time() - start_time > self.timeout:
                break
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Оптимізація завершена за {elapsed_time:.2f} секунд.")
            print(f"Знайдено найкращий стан з значенням {best_value} за {total_iterations} ітерацій.")
        
        return best_state, best_value, history

@OptimizerFactory.register("steepest_descent")
class SteepestDescent(HillClimbing):
    """
    Реалізація алгоритму Steepest Descent (градієнтного спуску).
    
    Steepest Descent - це варіант Hill Climbing, який завжди вибирає
    найкращий крок серед усіх можливих сусідів.
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        max_no_improvement: int = 50,
        timeout: float = 60.0,
        verbose: bool = False
    ):
        super().__init__(
            max_iterations=max_iterations,
            max_no_improvement=max_no_improvement,
            timeout=timeout,
            random_restarts=1,  # Steepest Descent зазвичай не використовує перезапуски
            verbose=verbose
        )
    
    def optimize(
        self,
        initial_state: T,
        objective_function: Callable[[T], float],
        neighborhood_function: Callable[[T], List[T]],
        is_maximizing: bool = False,
        **kwargs
    ) -> Tuple[T, float, List[Tuple[int, float]]]:
        """
        Оптимізує стан за допомогою алгоритму Steepest Descent.
        
        Steepest Descent завжди вибирає найкращий крок серед усіх можливих,
        тому ми перевіряємо всіх сусідів і вибираємо найкращого.
        """
        # Використовуємо базову реалізацію Hill Climbing
        return super().optimize(
            initial_state=initial_state,
            objective_function=objective_function,
            neighborhood_function=neighborhood_function,
            is_maximizing=is_maximizing,
            **kwargs
        )

@OptimizerFactory.register("random_search")
class RandomSearch(Optimizer[T, Tuple[T, float, List[Tuple[int, float]]]]):
    """Реалізація алгоритму випадкового пошуку."""
    
    def __init__(
        self,
        max_iterations: int = 1000,
        timeout: float = 60.0,
        verbose: bool = False
    ):
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.verbose = verbose
    
    def optimize(
        self,
        initial_state: T,
        objective_function: Callable[[T], float],
        neighborhood_function: Callable[[T], List[T]],
        is_maximizing: bool = False,
        **kwargs
    ) -> Tuple[T, float, List[Tuple[int, float]]]:
        """
        Оптимізує стан за допомогою алгоритму випадкового пошуку.
        
        Random Search генерує випадкові стани і вибирає найкращий з них.
        """
        best_state = copy.deepcopy(initial_state)
        best_value = objective_function(best_state)
        
        if self.verbose:
            print(f"Початковий стан має значення функції: {best_value}")
        
        history = [(0, best_value)]
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Перевіряємо таймаут
            if time.time() - start_time > self.timeout:
                if self.verbose:
                    print(f"Досягнуто ліміт часу ({self.timeout} секунд). Зупиняємо оптимізацію.")
                break
            
            # Генеруємо випадковий стан
            neighbors = neighborhood_function(best_state)
            
            if not neighbors:
                if self.verbose:
                    print("Немає сусідніх станів. Зупиняємо оптимізацію.")
                break
            
            # Вибираємо випадкового сусіда
            random_idx = random.randint(0, len(neighbors) - 1)
            random_state = neighbors[random_idx]
            random_value = objective_function(random_state)
            
            # Оновлюємо найкращий стан, якщо знайдено кращий
            if (is_maximizing and random_value > best_value) or \
               (not is_maximizing and random_value < best_value):
                best_state = copy.deepcopy(random_state)
                best_value = random_value
                if self.verbose:
                    print(f"Ітерація {iteration}: знайдено новий найкращий стан з значенням {best_value}")
            
            # Зберігаємо історію оцінок
            if iteration % 10 == 0:
                history.append((iteration, best_value))
        
        # Зберігаємо останню точку історії
        if (self.max_iterations - 1) % 10 != 0:
            history.append((self.max_iterations - 1, best_value))
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Оптимізація завершена за {elapsed_time:.2f} секунд.")
            print(f"Знайдено найкращий стан з значенням {best_value} за {self.max_iterations} ітерацій.")
        
        return best_state, best_value, history
