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


@OptimizerFactory.register("genetic_algorithm")
class GeneticAlgorithm(SwarmIntelligence):
    """Реалізація генетичного алгоритму для оптимізації логістичних задач."""
    
    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_ratio: float = 0.1,  # Частка кращих особин, які автоматично переходять в наступне покоління
        selection_method: str = "tournament",  # tournament, roulette, rank
        tournament_size: int = 5,  # Для турнірної селекції
        timeout: float = 60.0,
        verbose: bool = False
    ):
        """
        Ініціалізує генетичний алгоритм.
        
        Args:
            population_size: Розмір популяції
            max_generations: Максимальна кількість поколінь
            mutation_rate: Ймовірність мутації
            crossover_rate: Ймовірність схрещування
            elitism_ratio: Частка кращих особин, які автоматично переходять в наступне покоління
            selection_method: Метод селекції (tournament, roulette, rank)
            tournament_size: Розмір турніру для турнірної селекції
            timeout: Максимальний час виконання у секундах
            verbose: Режим детального логування
        """
        super().__init__(max_iterations=max_generations, population_size=population_size, timeout=timeout)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.verbose = verbose
    
    def optimize(
        self,
        initial_state: T,
        objective_function: Callable[[T], float],
        generate_individual: Callable[[], T],
        crossover_function: Callable[[T, T], Tuple[T, T]],
        mutation_function: Callable[[T], T],
        is_maximizing: bool = False,
        **kwargs
    ) -> Tuple[T, float, List[Tuple[int, float]]]:
        """
        Оптимізує стан за допомогою генетичного алгоритму.
        
        Args:
            initial_state: Початковий стан (може бути використаний для ініціалізації популяції)
            objective_function: Функція оцінки стану (функція пристосованості)
            generate_individual: Функція для генерації нового індивіда
            crossover_function: Функція схрещування двох особин
            mutation_function: Функція мутації особини
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
            
            # Елітизм: зберігаємо найкращі особини
            elite_count = int(self.population_size * self.elitism_ratio)
            
            # Сортуємо індекси популяції за пристосованістю
            if is_maximizing:
                sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)
            else:
                sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
            
            elite_indices = sorted_indices[:elite_count]
            elite_individuals = [copy.deepcopy(population[i]) for i in elite_indices]
            
            # Створюємо нову популяцію
            new_population = []
            
            # Додаємо елітні особини
            new_population.extend(elite_individuals)
            
            # Заповнюємо решту популяції нащадками
            while len(new_population) < self.population_size:
                # Вибираємо батьків
                if self.selection_method == "tournament":
                    parent1 = self._tournament_selection(population, fitness_values, is_maximizing)
                    parent2 = self._tournament_selection(population, fitness_values, is_maximizing)
                elif self.selection_method == "roulette":
                    parent1 = self._roulette_selection(population, fitness_values, is_maximizing)
                    parent2 = self._roulette_selection(population, fitness_values, is_maximizing)
                else:  # rank
                    parent1 = self._rank_selection(population, fitness_values, is_maximizing)
                    parent2 = self._rank_selection(population, fitness_values, is_maximizing)
                
                # Схрещування
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = crossover_function(parent1, parent2)
                else:
                    offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # Мутація
                if random.random() < self.mutation_rate:
                    offspring1 = mutation_function(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = mutation_function(offspring2)
                
                # Додаємо нащадків до нової популяції
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            # Оновлюємо популяцію
            population = new_population
            
            # Оцінюємо нову популяцію
            fitness_values = [objective_function(ind) for ind in population]
            
            # Знаходимо найкращу особину в цьому поколінні
            if is_maximizing:
                current_best_idx = max(range(len(fitness_values)), key=lambda i: fitness_values[i])
            else:
                current_best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
                
            current_best = population[current_best_idx]
            current_best_fitness = fitness_values[current_best_idx]
            
            # Оновлюємо найкращу знайдену особину
            if (is_maximizing and current_best_fitness > best_fitness) or \
               (not is_maximizing and current_best_fitness < best_fitness):
                best_individual = copy.deepcopy(current_best)
                best_fitness = current_best_fitness
                
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
    
    def _tournament_selection(self, population, fitness_values, is_maximizing):
        """Турнірна селекція."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        
        if is_maximizing:
            winner_idx = max(tournament_indices, key=lambda i: fitness_values[i])
        else:
            winner_idx = min(tournament_indices, key=lambda i: fitness_values[i])
            
        return copy.deepcopy(population[winner_idx])
    
    def _roulette_selection(self, population, fitness_values, is_maximizing):
        """Селекція методом рулетки."""
        # Для максимізації використовуємо значення пристосованості безпосередньо
        # Для мінімізації використовуємо обернені значення
        if is_maximizing:
            weights = fitness_values
        else:
            # Зміщуємо значення, щоб уникнути від'ємних або нульових значень
            min_fitness = min(fitness_values)
            weights = [max(1.0, min_fitness * 2 - f) for f in fitness_values]
        
        # Вибираємо індекс з імовірністю, пропорційною вазі
        total_weight = sum(weights)
        
        if total_weight == 0:
            # Якщо всі ваги нульові, використовуємо рівномірний розподіл
            return copy.deepcopy(random.choice(population))
            
        pick = random.uniform(0, total_weight)
        current = 0
        
        for i, weight in enumerate(weights):
            current += weight
            if current > pick:
                return copy.deepcopy(population[i])
                
        # На випадок помилки округлення
        return copy.deepcopy(population[-1])
    
    def _rank_selection(self, population, fitness_values, is_maximizing):
        """Рангова селекція."""
        # Сортуємо індекси за пристосованістю
        if is_maximizing:
            sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)
        else:
            sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
            
        # Призначаємо ранги (найкращий отримує найвищий ранг)
        ranks = list(range(1, len(population) + 1))
        
        # Вибираємо індекс з імовірністю, пропорційною рангу
        total_rank = sum(ranks)
        pick = random.uniform(0, total_rank)
        current = 0
        
        for i, rank in enumerate(ranks):
            current += rank
            if current > pick:
                return copy.deepcopy(population[sorted_indices[i]])
                
        # На випадок помилки округлення
        return copy.deepcopy(population[sorted_indices[-1]])