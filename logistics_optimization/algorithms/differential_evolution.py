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


@OptimizerFactory.register("differential_evolution")
class DifferentialEvolution(SwarmIntelligence):
    """Реалізація алгоритму диференціальної еволюції для оптимізації логістичних задач."""
    
    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        F: float = 0.8,  # Фактор мутації [0, 2]
        CR: float = 0.5,  # Коефіцієнт рекомбінації [0, 1]
        strategy: str = "DE/rand/1/bin",  # Стратегія DE
        timeout: float = 60.0,
        verbose: bool = False
    ):
        """
        Ініціалізує алгоритм диференціальної еволюції.
        
        Args:
            population_size: Розмір популяції
            max_generations: Максимальна кількість поколінь
            F: Фактор мутації [0, 2]
            CR: Коефіцієнт рекомбінації [0, 1]
            strategy: Стратегія диференціальної еволюції (DE/rand/1/bin, DE/best/1/bin, DE/rand/2/bin, ...)
            timeout: Максимальний час виконання у секундах
            verbose: Режим детального логування
        """
        super().__init__(max_iterations=max_generations, population_size=population_size, timeout=timeout)
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.verbose = verbose
    
    def optimize(
        self,
        initial_state: T,
        objective_function: Callable[[T], float],
        generate_individual: Callable[[], T],
        recombination_function: Callable[[T, T, float], T],
        ensure_bounds_function: Callable[[T], T] = None,
        is_maximizing: bool = False,
        **kwargs
    ) -> Tuple[T, float, List[Tuple[int, float]]]:
        """
        Оптимізує стан за допомогою алгоритму диференціальної еволюції.
        
        Args:
            initial_state: Початковий стан (може бути використаний для ініціалізації популяції)
            objective_function: Функція оцінки стану (функція пристосованості)
            generate_individual: Функція для генерації нового індивіда
            recombination_function: Функція рекомбінації (створює нащадка з батьківської особини та донора)
            ensure_bounds_function: Функція для перевірки та корекції обмежень (може бути None)
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
            **kwargs: Додаткові параметри
            
        Returns:
            Tuple[T, float, List[Tuple[int, float]]]: 
                Оптимізований стан, його оцінка та історія оцінок
        """
        # Ініціалізуємо популяцію
        population = [generate_individual() for _ in range(self.population_size)]
        
        # Додаємо початковий стан до популяції (заміщуємо останнього індивіда)
        population[-1] = copy.deepcopy(initial_state)
        
        # Оцінюємо початкову популяцію
        fitness_values = [objective_function(ind) for ind in population]
        
        # Знаходимо найкращу особину
        if is_maximizing:
            best_idx = max(range(len(fitness_values)), key=lambda i: fitness_values[i])
        else:
            best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
            
        best_individual = copy.deepcopy(population[best_idx])
        best_fitness = fitness_values[best_idx]
        
        if self.verbose:
            print(f"Початкова популяція: найкраща пристосованість = {best_fitness}")
        
        history = [(0, best_fitness)]
        
        start_time = time.time()
        
        for generation in range(self.max_iterations):
            # Перевіряємо таймаут
            if time.time() - start_time > self.timeout:
                if self.verbose:
                    print(f"Досягнуто ліміт часу ({self.timeout} секунд). Зупиняємо оптимізацію.")
                break
            
            # Створюємо нову популяцію
            for i in range(self.population_size):
                # Вибираємо базового індивіда для мутації
                if "best" in self.strategy:
                    base_idx = best_idx
                else:  # "rand"
                    candidates = list(range(self.population_size))
                    candidates.remove(i)  # Виключаємо поточного індивіда
                    base_idx = random.choice(candidates)
                
                # Вибираємо випадкові індекси для мутації, відмінні від i та base_idx
                candidates = list(range(self.population_size))
                candidates.remove(i)
                if base_idx != i:  # Перевіряємо, щоб не видалити i двічі
                    candidates.remove(base_idx)
                
                # Кількість випадкових індексів залежить від стратегії
                if "/1/" in self.strategy:
                    num_rand = 2  # Для DE/x/1/x потрібно 2 випадкових індекси
                elif "/2/" in self.strategy:
                    num_rand = 4  # Для DE/x/2/x потрібно 4 випадкових індекси
                else:
                    num_rand = 2  # За замовчуванням
                
                # Вибираємо випадкові індекси
                rand_indices = random.sample(candidates, min(num_rand, len(candidates)))
                
                # Створюємо донора шляхом мутації
                if "/1/" in self.strategy:
                    donor = self._mutate_1(population[base_idx], population[rand_indices[0]], population[rand_indices[1]])
                elif "/2/" in self.strategy:
                    donor = self._mutate_2(population[base_idx], 
                                           population[rand_indices[0]], population[rand_indices[1]],
                                           population[rand_indices[2]], population[rand_indices[3]])
                else:
                    donor = self._mutate_1(population[base_idx], population[rand_indices[0]], population[rand_indices[1]])
                
                # Рекомбінація (схрещування)
                trial = recombination_function(population[i], donor, self.CR)
                
                # Перевіряємо обмеження, якщо потрібно
                if ensure_bounds_function:
                    trial = ensure_bounds_function(trial)
                
                # Оцінюємо нащадка
                trial_fitness = objective_function(trial)
                
                # Відбір: заміщуємо батька, якщо нащадок кращий
                if (is_maximizing and trial_fitness > fitness_values[i]) or \
                   (not is_maximizing and trial_fitness < fitness_values[i]):
                    population[i] = trial
                    fitness_values[i] = trial_fitness
                    
                    # Оновлюємо найкращу особину
                    if (is_maximizing and trial_fitness > best_fitness) or \
                       (not is_maximizing and trial_fitness < best_fitness):
                        best_individual = copy.deepcopy(trial)
                        best_fitness = trial_fitness
                        best_idx = i
                        
                        if self.verbose:
                            print(f"Покоління {generation}: знайдено нову найкращу особину з пристосованістю {best_fitness}")
            
            # Зберігаємо історію оцінок
            if generation % 5 == 0:  # Зберігаємо кожне 5-те покоління для економії пам'яті
                history.append((generation, best_fitness))
        
        # Додаємо останню точку до історії
        if history[-1][0] != self.max_iterations - 1:
            history.append((self.max_iterations - 1, best_fitness))
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Оптимізація завершена за {elapsed_time:.2f} секунд.")
            print(f"Знайдено найкращу особину з пристосованістю {best_fitness} за {generation} поколінь.")
        
        return best_individual, best_fitness, history
    
    def _mutate_1(self, base, rand1, rand2):
        """Мутація DE/x/1: base + F * (rand1 - rand2)."""
        # Реалізація залежить від типу даних; це лише шаблон
        # У випадку з числовими векторами це буде:
        # base + self.F * (rand1 - rand2)
        # Але для інших типів даних (наприклад, маршрути) потрібна спеціальна реалізація
        
        if hasattr(base, "__add__") and hasattr(rand1, "__sub__") and isinstance(base, (list, tuple, np.ndarray)):
            result = base + self.F * (rand1 - rand2)
            return result
        else:
            # Для інших типів даних повертаємо копію бази
            return copy.deepcopy(base)
    
    def _mutate_2(self, base, rand1, rand2, rand3, rand4):
        """Мутація DE/x/2: base + F * (rand1 + rand2 - rand3 - rand4)."""
        if hasattr(base, "__add__") and hasattr(rand1, "__sub__") and isinstance(base, (list, tuple, np.ndarray)):
            result = base + self.F * (rand1 + rand2 - rand3 - rand4)
            return result
        else:
            # Для інших типів даних повертаємо копію бази
            return copy.deepcopy(base)