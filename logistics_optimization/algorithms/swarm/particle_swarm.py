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

@OptimizerFactory.register("particle_swarm")
class ParticleSwarmOptimization(SwarmIntelligence):
    """Реалізація алгоритму оптимізації роєм частинок (PSO) для логістичних задач."""
    
    def __init__(
        self,
        swarm_size: int = 50,
        max_iterations: int = 100,
        inertia_weight: float = 0.5,
        cognitive_coefficient: float = 1.5,  # c1
        social_coefficient: float = 1.5,     # c2
        timeout: float = 60.0,
        verbose: bool = False
    ):
        """
        Ініціалізує алгоритм оптимізації роєм частинок.
        
        Args:
            swarm_size: Розмір рою (кількість частинок)
            max_iterations: Максимальна кількість ітерацій
            inertia_weight: Вага інерції (впливає на швидкість руху частинок)
            cognitive_coefficient: Когнітивний коефіцієнт (c1, вплив особистого найкращого положення)
            social_coefficient: Соціальний коефіцієнт (c2, вплив глобального найкращого положення)
            timeout: Максимальний час виконання у секундах
            verbose: Режим детального логування
        """
        super().__init__(max_iterations=max_iterations, population_size=swarm_size, timeout=timeout)
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.verbose = verbose
    
    def optimize(
        self,
        initial_state: T,
        objective_function: Callable[[T], float],
        generate_position: Callable[[], T],
        update_position: Callable[[T, Any], T],
        generate_velocity: Callable[[], Any],
        update_velocity: Callable[[Any, T, T, T, float, float, float], Any],
        ensure_bounds_function: Callable[[T], T] = None,
        is_maximizing: bool = False,
        **kwargs
    ) -> Tuple[T, float, List[Tuple[int, float]]]:
        """
        Оптимізує стан за допомогою алгоритму оптимізації роєм частинок.
        
        Args:
            initial_state: Початковий стан (може бути використаний для ініціалізації положення частинки)
            objective_function: Функція оцінки стану (функція пристосованості)
            generate_position: Функція для генерації нового положення частинки
            update_position: Функція для оновлення положення частинки
            generate_velocity: Функція для генерації початкової швидкості частинки
            update_velocity: Функція для оновлення швидкості частинки
            ensure_bounds_function: Функція для перевірки та корекції обмежень (може бути None)
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
            **kwargs: Додаткові параметри
            
        Returns:
            Tuple[T, float, List[Tuple[int, float]]]: 
                Оптимізований стан, його оцінка та історія оцінок
        """
        # Ініціалізуємо рій
        positions = [generate_position() for _ in range(self.population_size)]
        velocities = [generate_velocity() for _ in range(self.population_size)]
        
        # Додаємо початковий стан до рою (заміщуємо останню частинку)
        positions[-1] = copy.deepcopy(initial_state)
        
        # Оцінюємо початкові положення
        fitness_values = [objective_function(pos) for pos in positions]
        
        # Ініціалізуємо особисті найкращі положення
        personal_best_positions = copy.deepcopy(positions)
        personal_best_fitness = copy.deepcopy(fitness_values)
        
        # Знаходимо глобальне найкраще положення
        if is_maximizing:
            global_best_idx = max(range(len(fitness_values)), key=lambda i: fitness_values[i])
        else:
            global_best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
            
        global_best_position = copy.deepcopy(positions[global_best_idx])
        global_best_fitness = fitness_values[global_best_idx]
        
        if self.verbose:
            print(f"Початковий рій: найкраща пристосованість = {global_best_fitness}")
        
        history = [(0, global_best_fitness)]
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Перевіряємо таймаут
            if time.time() - start_time > self.timeout:
                if self.verbose:
                    print(f"Досягнуто ліміт часу ({self.timeout} секунд). Зупиняємо оптимізацію.")
                break
            
            # Оновлюємо положення і швидкості частинок
            for i in range(self.population_size):
                # Оновлюємо швидкість
                velocities[i] = update_velocity(
                    velocities[i],
                    positions[i],
                    personal_best_positions[i],
                    global_best_position,
                    self.inertia_weight,
                    self.cognitive_coefficient,
                    self.social_coefficient
                )
                
                # Оновлюємо положення
                positions[i] = update_position(positions[i], velocities[i])
                
                # Перевіряємо обмеження, якщо потрібно
                if ensure_bounds_function:
                    positions[i] = ensure_bounds_function(positions[i])
                
                # Оцінюємо нове положення
                fitness = objective_function(positions[i])
                
                # Оновлюємо особисте найкраще положення
                if (is_maximizing and fitness > personal_best_fitness[i]) or \
                   (not is_maximizing and fitness < personal_best_fitness[i]):
                    personal_best_positions[i] = copy.deepcopy(positions[i])
                    personal_best_fitness[i] = fitness
                    
                    # Оновлюємо глобальне найкраще положення
                    if (is_maximizing and fitness > global_best_fitness) or \
                       (not is_maximizing and fitness < global_best_fitness):
                        global_best_position = copy.deepcopy(positions[i])
                        global_best_fitness = fitness
                        
                        if self.verbose:
                            print(f"Ітерація {iteration}: знайдено нове найкраще положення з пристосованістю {global_best_fitness}")
            
            # Зберігаємо історію оцінок
            if iteration % 5 == 0:  # Зберігаємо кожну 5-ту ітерацію для економії пам'яті
                history.append((iteration, global_best_fitness))
        
        # Додаємо останню точку до історії
        if history[-1][0] != self.max_iterations - 1:
            history.append((self.max_iterations - 1, global_best_fitness))
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Оптимізація завершена за {elapsed_time:.2f} секунд.")
            print(f"Знайдено найкраще положення з пристосованістю {global_best_fitness} за {iteration} ітерацій.")
        
        return global_best_position, global_best_fitness, history