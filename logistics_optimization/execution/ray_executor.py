import ray
import time
import copy
import logging
import numpy as np
import os
from typing import List, Dict, Tuple, Callable, Any, TypeVar, Generic, Optional, Union
from concurrent.futures import ProcessPoolExecutor
from core.optimizer import Optimizer, OptimizerFactory

T = TypeVar('T')  # Тип початкового стану
R = TypeVar('R')  # Тип результату

class RayExecutor:
    """Клас для паралельного виконання алгоритмів оптимізації з використанням Ray."""
    
    def __init__(
        self, 
        use_local: bool = False, 
        num_cpus: int = None,
        address: str = None,
        initialize_ray: bool = True,
        verbose: bool = False
    ):
        """
        Ініціалізує Ray-виконавця.
        
        Args:
            use_local: Якщо True, виконує операції локально (без Ray)
            num_cpus: Кількість CPU для використання, якщо None - використовує всі доступні
            address: Адреса кластера Ray для підключення
            initialize_ray: Якщо True, ініціалізує Ray під час створення об'єкта
            verbose: Режим детального логування
        """
        self.use_local = use_local
        self.num_cpus = num_cpus
        self.address = address
        self.verbose = verbose
        
        if initialize_ray and not use_local and not ray.is_initialized():
            self._initialize_ray()
    
    def _initialize_ray(self):
        """Ініціалізує Ray з заданими параметрами."""
        try:
            if self.address:
                # Підключаємось до існуючого кластера
                if self.verbose:
                    print(f"Підключення до кластера Ray за адресою: {self.address}")
                ray.init(address=self.address)
            else:
                # Створюємо локальний кластер
                if self.verbose:
                    print(f"Ініціалізація Ray з {self.num_cpus or 'всіма доступними'} CPU")
                ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True)
            
            if self.verbose:
                print(f"Ray initialized: {ray.cluster_resources()}")
        except Exception as e:
            print(f"Помилка ініціалізації Ray: {str(e)}")
            self.use_local = True
    
    def parallel_map(self, func: Callable[[T], R], items: List[T]) -> List[R]:
        """
        Паралельно виконує функцію для кожного елемента списку.
        
        Args:
            func: Функція для виконання
            items: Список вхідних даних
            
        Returns:
            List[R]: Список результатів
        """
        if self.use_local:
            # Для локального виконання можна використати ProcessPoolExecutor
            # для паралелізму без Ray
            with ProcessPoolExecutor(max_workers=self.num_cpus) as executor:
                return list(executor.map(func, items))
        
        # Конвертуємо функцію в Ray-задачу
        @ray.remote
        def ray_func(item):
            return func(item)
        
        # Відправляємо задачі на виконання
        futures = [ray_func.remote(item) for item in items]
        
        # Отримуємо результати
        return ray.get(futures)
    
    def parallel_optimize(
        self,
        algorithm_class: Union[type, str],
        algorithm_params: Dict[str, Any],
        initial_states: List[T],
        objective_function: Callable[[T], float],
        neighborhood_function: Callable[[T], List[T]],
        is_maximizing: bool = False,
        **kwargs
    ) -> Tuple[T, float, List[Tuple[int, float]]]:
        """
        Паралельно запускає алгоритм оптимізації з різними початковими станами.
        
        Args:
            algorithm_class: Клас алгоритму оптимізації або його назва для OptimizerFactory
            algorithm_params: Параметри алгоритму оптимізації
            initial_states: Список початкових станів
            objective_function: Функція оцінки стану
            neighborhood_function: Функція, що генерує сусідні стани
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
            **kwargs: Додаткові параметри для методу optimize алгоритму
            
        Returns:
            Tuple[T, float, List[Tuple[int, float]]]: 
                Найкращий знайдений стан, його оцінка та історія оцінок
        """
        # Функція для виконання оптимізації з одним початковим станом
        def optimize_single(initial_state):
            # Створюємо екземпляр алгоритму
            if isinstance(algorithm_class, str):
                algorithm = OptimizerFactory.create_optimizer(algorithm_class, algorithm_params)
            else:
                algorithm = algorithm_class(**algorithm_params)
            
            # Запускаємо оптимізацію
            best_state, best_value, history = algorithm.optimize(
                initial_state=initial_state,
                objective_function=objective_function,
                neighborhood_function=neighborhood_function,
                is_maximizing=is_maximizing,
                **kwargs
            )
            
            return best_state, best_value, history
        
        # Виконуємо оптимізацію для кожного початкового стану
        start_time = time.time()
        
        if self.verbose:
            print(f"Запуск паралельної оптимізації з {len(initial_states)} початковими станами")
        
        # Паралельно запускаємо оптимізацію для кожного початкового стану
        results = self.parallel_map(optimize_single, initial_states)
        
        # Вибираємо найкращий результат
        if is_maximizing:
            best_idx = max(range(len(results)), key=lambda i: results[i][1])
        else:
            best_idx = min(range(len(results)), key=lambda i: results[i][1])
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Паралельна оптимізація завершена за {elapsed_time:.2f} секунд")
            print(f"Найкращий результат (з {len(results)} запусків): {results[best_idx][1]}")
        
        return results[best_idx]
    
    def batch_simulate(
        self,
        simulation_func: Callable[[Dict[str, Any]], Dict[str, Any]],
        parameter_sets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Паралельно виконує симуляцію з різними наборами параметрів.
        
        Args:
            simulation_func: Функція симуляції
            parameter_sets: Список наборів параметрів
            
        Returns:
            List[Dict[str, Any]]: Список результатів симуляції
        """
        return self.parallel_map(simulation_func, parameter_sets)
    
    def grid_search(
        self,
        algorithm_name: str,
        base_params: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        initial_state: T,
        objective_function: Callable[[T], float],
        neighborhood_function: Callable[[T], List[T]],
        is_maximizing: bool = False,
        **kwargs
    ) -> Tuple[Dict[str, Any], T, float]:
        """
        Виконує пошук по сітці для параметрів алгоритму оптимізації.
        
        Args:
            algorithm_name: Назва алгоритму для OptimizerFactory
            base_params: Базові параметри алгоритму
            param_grid: Сітка параметрів для перебору
            initial_state: Початковий стан
            objective_function: Функція оцінки стану
            neighborhood_function: Функція, що генерує сусідні стани
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
            **kwargs: Додаткові параметри для методу optimize
            
        Returns:
            Tuple[Dict[str, Any], T, float]: 
                Найкращі параметри, найкращий стан і його оцінка
        """
        # Генеруємо всі комбінації параметрів
        param_combinations = self._generate_param_combinations(param_grid)
        
        if self.verbose:
            print(f"Запуск пошуку по сітці з {len(param_combinations)} комбінаціями параметрів")
        
        # Функція для виконання оптимізації з одним набором параметрів
        def optimize_with_params(params):
            full_params = {**base_params, **params}
            algorithm = OptimizerFactory.create_optimizer(algorithm_name, full_params)
            
            best_state, best_value, _ = algorithm.optimize(
                initial_state=copy.deepcopy(initial_state),
                objective_function=objective_function,
                neighborhood_function=neighborhood_function,
                is_maximizing=is_maximizing,
                **kwargs
            )
            
            return params, best_state, best_value
        
        # Паралельно запускаємо оптимізацію для кожного набору параметрів
        results = self.parallel_map(optimize_with_params, param_combinations)
        
        # Вибираємо найкращий результат
        if is_maximizing:
            best_idx = max(range(len(results)), key=lambda i: results[i][2])
        else:
            best_idx = min(range(len(results)), key=lambda i: results[i][2])
        
        if self.verbose:
            print(f"Найкращі параметри: {results[best_idx][0]}")
            print(f"Найкраще значення: {results[best_idx][2]}")
        
        return results[best_idx]
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]], current_params=None, current_key_idx=0):
        """Генерує всі комбінації параметрів з сітки."""
        if current_params is None:
            current_params = {}
        
        param_keys = list(param_grid.keys())
        
        if current_key_idx >= len(param_keys):
            return [current_params.copy()]
        
        current_key = param_keys[current_key_idx]
        combinations = []
        
        for param_value in param_grid[current_key]:
            current_params[current_key] = param_value
            combinations.extend(
                self._generate_param_combinations(
                    param_grid, current_params, current_key_idx + 1
                )
            )
        
        return combinations