from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Callable, Dict, TypeVar, Generic, Optional

T = TypeVar('T')  # Тип рішення (наприклад, маршрут або розподіл посилок)
R = TypeVar('R')  # Тип результату

class Optimizer(ABC, Generic[T, R]):
    """Абстрактний базовий клас для всіх алгоритмів оптимізації."""
    
    @abstractmethod
    def optimize(
        self,
        initial_state: T,
        objective_function: Callable[[T], float],
        neighborhood_function: Callable[[T], List[T]],
        is_maximizing: bool = False,
        **kwargs
    ) -> R:
        """
        Метод оптимізації, який повинні реалізувати всі підкласи.
        
        Args:
            initial_state: Початковий стан
            objective_function: Функція оцінки стану
            neighborhood_function: Функція, що генерує сусідні стани
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
            **kwargs: Додаткові параметри, специфічні для конкретного алгоритму
            
        Returns:
            R: Результат оптимізації
        """
        pass


class MetaHeuristic(Optimizer):
    """Базовий клас для метаевристичних алгоритмів."""
    
    def __init__(self, max_iterations: int = 1000, timeout: float = 60.0):
        self.max_iterations = max_iterations
        self.timeout = timeout
        
    def log_progress(self, iteration: int, best_value: float, current_value: float):
        """Логує прогрес оптимізації."""
        pass


class SwarmIntelligence(MetaHeuristic):
    """Базовий клас для алгоритмів ройового інтелекту."""
    
    def __init__(self, max_iterations: int = 1000, population_size: int = 20, timeout: float = 60.0):
        super().__init__(max_iterations, timeout)
        self.population_size = population_size


class OptimizerFactory:
    """Фабрика для створення оптимізаторів."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name):
        """
        Декоратор для реєстрації класів оптимізаторів.
        
        Args:
            name: Назва оптимізатора
            
        Returns:
            Функція-декоратор
        """
        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    @classmethod
    def create_optimizer(cls, name: str, config: Dict[str, Any] = None) -> Optimizer:
        """
        Створює екземпляр оптимізатора за назвою.
        
        Args:
            name: Назва оптимізатора
            config: Конфігурація оптимізатора
            
        Returns:
            Optimizer: Екземпляр оптимізатора
        """
        if name not in cls._registry:
            raise ValueError(f"Оптимізатор з назвою {name} не зареєстрований")
        
        optimizer_class = cls._registry[name]
        
        if config is None:
            config = {}
            
        return optimizer_class(**config)