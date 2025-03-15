import numpy as np
import random
import time
import copy
import logging
from typing import List, Tuple, Dict, Callable, TypeVar, Any
from core.optimizer import Optimizer, OptimizerFactory

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

T = TypeVar('T')  # Тип стану (наприклад, маршрут або розподіл посилок)

@OptimizerFactory.register("simulated_annealing")
class SimulatedAnnealing(Optimizer[T, Tuple[T, float, List[Tuple[int, float]]]]):
    """Імплементація алгоритму імітації відпалу для оптимізації логістичних маршрутів."""
    
    def __init__(
        self, 
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        min_temp: float = 0.1,
        max_iterations: int = 1000,
        reset_temp_schedule: Dict[float, float] = None,  # Графік скидання температури
        timeout: float = 60.0,
        verbose: bool = False
    ):
        """
        Ініціалізує алгоритм імітації відпалу.
        
        Args:
            initial_temp: Початкова температура
            cooling_rate: Коефіцієнт охолодження (0 < cooling_rate < 1)
            min_temp: Мінімальна температура
            max_iterations: Максимальна кількість ітерацій
            reset_temp_schedule: Графік скидання температури (ключ - відносна кількість ітерацій, значення - температура)
                                 Приклад: {0.5: 50.0, 0.8: 20.0} - скидання температури до 50 на 50% ітерацій і до 20 на 80%
            timeout: Максимальний час виконання у секундах
            verbose: Режим детального логування
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.reset_temp_schedule = reset_temp_schedule
        self.timeout = timeout
        self.verbose = verbose
        
    def optimize(
        self, 
        initial_solution: T,
        objective_function: Callable[[T], float],
        neighborhood_function: Callable[[T], T],
        is_maximizing: bool = False,
        fixed_points: List[int] = None,  # Індекси пунктів, які не можна змінювати
        **kwargs
    ) -> Tuple[T, float, List[Tuple[int, float]]]:
        """
        Оптимізує маршрут за допомогою імітації відпалу.
        
        Args:
            initial_solution: Початковий маршрут (список ID вузлів)
            objective_function: Функція оцінки вартості маршруту
            neighborhood_function: Функція, що генерує новий стан в околиці поточного
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
            fixed_points: Індекси вузлів, які мають залишатися на своїх місцях
            **kwargs: Додаткові параметри
            
        Returns:
            Tuple[T, float, List[Tuple[int, float]]]: 
                Оптимізований маршрут, його вартість та історія вартості
        """
        current_solution = copy.deepcopy(initial_solution)
        current_cost = objective_function(current_solution)
        
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        if self.verbose:
            print(f"Початковий стан має значення функції: {best_cost}")
        
        temperature = self.initial_temp
        cost_history = [(0, current_cost)]
        
        start_time = time.time()
        iteration = 0
        
        while temperature > self.min_temp and iteration < self.max_iterations:
            # Перевіряємо таймаут
            if time.time() - start_time > self.timeout:
                if self.verbose:
                    print(f"Досягнуто ліміт часу ({self.timeout} секунд). Зупиняємо оптимізацію.")
                break
            
            # Генеруємо новий стан
            new_solution = neighborhood_function(current_solution)
            
            # Переконуємося, що фіксовані точки залишаються на своїх місцях
            if fixed_points and hasattr(new_solution, '__getitem__') and hasattr(new_solution, '__setitem__'):
                for idx in fixed_points:
                    if idx < len(current_solution) and idx < len(new_solution):
                        new_solution[idx] = current_solution[idx]
            
            # Оцінюємо новий стан
            new_cost = objective_function(new_solution)
            
            # Обчислюємо зміну вартості
            cost_diff = new_cost - current_cost
            
            # Обертаємо різницю вартості для максимізації
            if is_maximizing:
                cost_diff = -cost_diff
            
            # Приймаємо рішення про перехід до нового стану
            if cost_diff < 0:  # Якщо новий стан кращий
                current_solution = new_solution
                current_cost = new_cost
                
                # Перевіряємо, чи це найкращий знайдений стан
                if (not is_maximizing and current_cost < best_cost) or (is_maximizing and current_cost > best_cost):
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost
                    if self.verbose:
                        print(f"Ітерація {iteration}: знайдено новий найкращий стан з значенням {best_cost}")
            else:
                # Якщо новий стан гірший, все одно приймаємо його з деякою ймовірністю
                acceptance_probability = np.exp(-cost_diff / temperature)
                if random.random() < acceptance_probability:
                    current_solution = new_solution
                    current_cost = new_cost
                    if self.verbose and iteration % 100 == 0:
                        print(f"Ітерація {iteration}: прийнято гірший стан з значенням {current_cost} (p={acceptance_probability:.4f})")
            
            # Перевіряємо графік скидання температури
            if self.reset_temp_schedule:
                rel_iteration = iteration / self.max_iterations
                for threshold, temp in sorted(self.reset_temp_schedule.items()):
                    if rel_iteration >= threshold and temperature < temp:
                        temperature = temp
                        if self.verbose:
                            print(f"Ітерація {iteration}: скинуто температуру до {temperature}")
                        break
            
            # Охолоджуємо систему
            temperature *= self.cooling_rate
            iteration += 1
            
            # Зберігаємо історію вартості
            if iteration % 10 == 0:  # Зберігаємо кожну 10-ту точку для економії пам'яті
                cost_history.append((iteration, current_cost))
        
        # Додаємо останню точку до історії
        if cost_history[-1][0] != iteration:
            cost_history.append((iteration, current_cost))
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Оптимізація завершена за {elapsed_time:.2f} секунд.")
            print(f"Знайдено найкращий стан з значенням {best_cost} за {iteration} ітерацій.")
        
        return best_solution, best_cost, cost_history

# Приклади функцій околиці для задачі маршрутизації

def swap_nodes(route: List) -> List:
    """Міняє місцями два випадкові вузли в маршруті."""
    result = copy.deepcopy(route)
    if len(result) <= 2:
        return result
    
    idx1, idx2 = random.sample(range(1, len(result) - 1), 2)  # Не міняємо початковий і кінцевий вузли
    result[idx1], result[idx2] = result[idx2], result[idx1]
    return result

def insert_node(route: List) -> List:
    """Вставляє випадковий вузол в іншу позицію маршруту."""
    result = copy.deepcopy(route)
    if len(result) <= 2:
        return result
    
    # Вибираємо випадковий вузол для переміщення, крім початкового і кінцевого
    src_idx = random.randint(1, len(result) - 2)
    
    # Вибираємо нову позицію, крім початкової та кінцевої вузлів
    dest_idx = random.randint(1, len(result) - 2)
    while dest_idx == src_idx:
        dest_idx = random.randint(1, len(result) - 2)
    
    # Видаляємо вузол з вихідної позиції
    node = result.pop(src_idx)
    
    # Вставляємо його в нову позицію
    if dest_idx > src_idx:
        dest_idx -= 1  # Корегуємо індекс, оскільки масив став на 1 коротше
    result.insert(dest_idx, node)
    
    return result

def reverse_segment(route: List) -> List:
    """Обертає сегмент маршруту."""
    result = copy.deepcopy(route)
    if len(result) <= 3:
        return result
    
    # Вибираємо випадковий сегмент, крім першого і останнього вузлів
    start = random.randint(1, len(result) - 3)
    end = random.randint(start + 1, len(result) - 2)
    
    # Обертаємо сегмент
    result[start:end+1] = reversed(result[start:end+1])
    
    return result

def mixed_neighborhood(route: List) -> List:
    """Застосовує випадково вибрану функцію околиці."""
    choice = random.randint(0, 2)
    if choice == 0:
        return swap_nodes(route)
    elif choice == 1:
        return insert_node(route)
    else:
        return reverse_segment(route)

# Функція-обгортка для використання з класом SimulatedAnnealing
def get_route_cost_function(network: 'LogisticsNetwork'):
    """
    Повертає функцію для обчислення вартості маршруту в логістичній мережі.
    
    Args:
        network: Об'єкт логістичної мережі
        
    Returns:
        Callable[[List[str]], float]: Функція обчислення вартості маршруту
    """
    def route_cost(route: List[str]) -> float:
        """Обчислює вартість маршруту в логістичній мережі."""
        if len(route) < 2:
            return float('inf')
        
        total_distance = 0
        for i in range(len(route) - 1):
            distance = network.get_distance(route[i], route[i+1])
            if distance == float('inf'):
                return float('inf')  # Немає зв'язку між вузлами
            total_distance += distance
        
        return total_distance
    
    return route_cost

def optimize_route(
    start_node: str,
    end_node: str,
    intermediate_nodes: List[str],
    network: 'LogisticsNetwork',
    vehicle_id: str = None,
    config: Dict[str, Any] = None
) -> Tuple[List[str], float]:
    """
    Оптимізує маршрут з початкового вузла до кінцевого через проміжні вузли.
    
    Args:
        start_node: ID початкового вузла
        end_node: ID кінцевого вузла
        intermediate_nodes: Список ID проміжних вузлів
        network: Об'єкт логістичної мережі
        vehicle_id: ID транспортного засобу (якщо потрібно враховувати обмеження)
        config: Конфігурація алгоритму імітації відпалу
        
    Returns:
        Tuple[List[str], float]: Оптимізований маршрут та його вартість
    """
    # Створюємо початковий маршрут
    initial_route = [start_node] + intermediate_nodes + [end_node]
    
    # Визначаємо функцію вартості
    cost_fn = get_route_cost_function(network)
    
    # Використовуємо стандартну конфігурацію, якщо не надано іншу
    if config is None:
        config = {
            "initial_temp": 100.0,
            "cooling_rate": 0.98,
            "min_temp": 0.01,
            "max_iterations": 10000,
            "verbose": False
        }
    
    # Створюємо оптимізатор за допомогою фабрики
    sa = OptimizerFactory.create_optimizer("simulated_annealing", config)
    
    # Оптимізуємо маршрут
    fixed_points = [0, len(initial_route) - 1]  # Фіксуємо початковий і кінцевий вузли
    optimized_route, cost, history = sa.optimize(
        initial_solution=initial_route,
        objective_function=cost_fn,
        neighborhood_function=mixed_neighborhood,
        fixed_points=fixed_points
    )
    
    return optimized_route, cost
