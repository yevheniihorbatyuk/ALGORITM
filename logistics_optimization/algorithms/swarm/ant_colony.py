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


@OptimizerFactory.register("ant_colony")
class AntColonyOptimization(SwarmIntelligence):
    """Реалізація алгоритму оптимізації мурашиною колонією (ACO) для логістичних задач."""
    
    def __init__(
        self,
        n_ants: int = 20,
        n_iterations: int = 100,
        alpha: float = 1.0,  # Вплив феромону
        beta: float = 2.0,   # Вплив евристичної інформації
        rho: float = 0.5,    # Швидкість випаровування феромону
        q0: float = 0.9,     # Параметр дослідження/експлуатації
        initial_pheromone: float = 0.1,
        min_pheromone: float = 0.001,
        max_pheromone: float = 1.0,
        timeout: float = 60.0,
        verbose: bool = False
    ):
        """
        Ініціалізує алгоритм оптимізації мурашиною колонією.
        
        Args:
            n_ants: Кількість мурах у колонії
            n_iterations: Кількість ітерацій алгоритму
            alpha: Вплив феромону на вибір шляху
            beta: Вплив евристичної інформації на вибір шляху
            rho: Швидкість випаровування феромону
            q0: Ймовірність використання "жадібного" правила вибору
            initial_pheromone: Початкова кількість феромону на ребрах
            min_pheromone: Мінімальна кількість феромону
            max_pheromone: Максимальна кількість феромону
            timeout: Максимальний час виконання у секундах
            verbose: Режим детального логування
        """
        super().__init__(max_iterations=n_iterations, population_size=n_ants, timeout=timeout)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.initial_pheromone = initial_pheromone
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        self.verbose = verbose
        self.pheromone = None  # Буде ініціалізовано під час оптимізації
    
    def optimize(
        self,
        distance_matrix: np.ndarray,
        start_node: int = 0,
        end_node: Optional[int] = None,
        construct_solution: Optional[Callable] = None,
        update_pheromone: Optional[Callable] = None,
        is_maximizing: bool = False,
        **kwargs
    ) -> Tuple[List[List[int]], float, List[float]]:
        """
        Запускає оптимізацію мурашиної колонії.
        
        Args:
            distance_matrix: Матриця відстаней між вузлами
            start_node: Початковий вузол для всіх маршрутів
            end_node: Кінцевий вузол для всіх маршрутів (якщо None, то дорівнює start_node)
            construct_solution: Функція для побудови рішення (якщо None, використовується внутрішня)
            update_pheromone: Функція для оновлення феромону (якщо None, використовується внутрішня)
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
            **kwargs: Додаткові параметри
            
        Returns:
            Tuple[List[List[int]], float, List[float]]: 
                Найкращі маршрути, їх загальна вартість та історія найкращих вартостей
        """
        n_nodes = distance_matrix.shape[0]
        
        if end_node is None:
            end_node = start_node
        
        # Ініціалізуємо матрицю феромонів
        self.pheromone = np.full((n_nodes, n_nodes), self.initial_pheromone)
        
        # Обчислюємо евристичну інформацію (1/відстань)
        heuristic = 1.0 / (distance_matrix + np.eye(n_nodes))  # Додаємо одиничну матрицю, щоб уникнути ділення на 0
        
        best_routes = None
        best_cost = float('inf')
        cost_history = []
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Перевіряємо таймаут
            if time.time() - start_time > self.timeout:
                if self.verbose:
                    print(f"Досягнуто ліміт часу ({self.timeout} секунд). Зупиняємо оптимізацію.")
                break
            
            all_routes = []
            all_costs = []
            
            for ant in range(self.population_size):
                # Будуємо рішення для кожної мурахи
                if construct_solution:
                    route, cost = construct_solution(
                        self.pheromone, heuristic, distance_matrix, start_node, end_node, 
                        self.alpha, self.beta, self.q0, **kwargs
                    )
                else:
                    route, cost = self._construct_solution(
                        self.pheromone, heuristic, distance_matrix, start_node, end_node, **kwargs
                    )
                
                all_routes.append(route)
                all_costs.append(cost)
            
            # Знаходимо найкращий результат на поточній ітерації
            if is_maximizing:
                best_ant = max(range(len(all_costs)), key=lambda i: all_costs[i])
            else:
                best_ant = min(range(len(all_costs)), key=lambda i: all_costs[i])
                
            iteration_best_route = all_routes[best_ant]
            iteration_best_cost = all_costs[best_ant]
            
            # Оновлюємо найкраще знайдене рішення
            if (is_maximizing and iteration_best_cost > best_cost) or \
               (not is_maximizing and iteration_best_cost < best_cost):
                best_routes = [iteration_best_route]
                best_cost = iteration_best_cost
                
                if self.verbose:
                    print(f"Ітерація {iteration}: знайдено новий найкращий маршрут з вартістю {best_cost}")
            
            # Оновлюємо феромони
            if update_pheromone:
                update_pheromone(
                    self.pheromone, all_routes, all_costs, self.rho, 
                    self.min_pheromone, self.max_pheromone, **kwargs
                )
            else:
                self._update_pheromone(all_routes, all_costs, **kwargs)
            
            # Зберігаємо історію найкращих вартостей
            cost_history.append(best_cost)
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Оптимізація завершена за {elapsed_time:.2f} секунд.")
            print(f"Знайдено найкращий маршрут з вартістю {best_cost} за {iteration} ітерацій.")
        
        return best_routes, best_cost, cost_history
    
    def _construct_solution(
        self,
        pheromone: np.ndarray,
        heuristic: np.ndarray,
        distance_matrix: np.ndarray,
        start_node: int,
        end_node: int,
        **kwargs
    ) -> Tuple[List[int], float]:
        """
        Будує рішення для однієї мурахи.
        
        Args:
            pheromone: Матриця феромонів
            heuristic: Матриця евристичної інформації
            distance_matrix: Матриця відстаней
            start_node: Початковий вузол
            end_node: Кінцевий вузол
            **kwargs: Додаткові параметри
            
        Returns:
            Tuple[List[int], float]: Маршрут і його вартість
        """
        n_nodes = distance_matrix.shape[0]
        
        # Початковий маршрут містить тільки початковий вузол
        route = [start_node]
        
        # Визначаємо, які вузли ще не відвідані
        unvisited = list(range(n_nodes))
        unvisited.remove(start_node)
        
        if start_node != end_node:
            # Якщо кінцевий вузол відрізняється від початкового, видаляємо його з невідвіданих
            # (ми додамо його в кінець маршруту)
            if end_node in unvisited:
                unvisited.remove(end_node)
        
        # Поточний вузол
        current_node = start_node
        
        while unvisited:
            # Обчислюємо ймовірності переходу в наступні вузли
            probabilities = np.zeros(len(unvisited))
            
            for i, next_node in enumerate(unvisited):
                # Обчислюємо привабливість вузла
                pheromone_value = pheromone[current_node, next_node]
                heuristic_value = heuristic[current_node, next_node]
                
                probabilities[i] = (pheromone_value ** self.alpha) * (heuristic_value ** self.beta)
            
            # Нормалізуємо ймовірності
            total = np.sum(probabilities)
            if total > 0:
                probabilities = probabilities / total
            
            # Вибираємо наступний вузол
            q = random.random()
            
            if q <= self.q0:
                # Використовуємо "жадібне" правило вибору
                next_idx = np.argmax(probabilities)
            else:
                # Використовуємо стохастичне правило вибору
                next_idx = np.random.choice(range(len(unvisited)), p=probabilities)
            
            next_node = unvisited[next_idx]
            
            # Оновлюємо маршрут
            route.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
        
        # Додаємо кінцевий вузол, якщо він відрізняється від поточного
        if current_node != end_node:
            route.append(end_node)
        
        # Обчислюємо вартість маршруту
        cost = 0
        for i in range(len(route) - 1):
            cost += distance_matrix[route[i], route[i + 1]]
        
        return route, cost
    
    def _update_pheromone(self, all_routes: List[List[int]], all_costs: List[float], **kwargs):
        """
        Оновлює матрицю феромонів.
        
        Args:
            all_routes: Маршрути всіх мурах
            all_costs: Вартості всіх маршрутів
            **kwargs: Додаткові параметри
        """
        # Випаровуємо феромони
        self.pheromone = (1 - self.rho) * self.pheromone
        
        # Додаємо феромони на шляхи мурах
        for route, cost in zip(all_routes, all_costs):
            # Кількість феромону, яку додаємо, обернено пропорційна вартості маршруту
            delta = 1.0 / max(0.1, cost)  # Запобігаємо діленню на 0
            
            for i in range(len(route) - 1):
                from_node, to_node = route[i], route[i + 1]
                self.pheromone[from_node, to_node] += delta
                self.pheromone[to_node, from_node] += delta  # Для неорієнтованого графа
        
        # Обмежуємо кількість феромону
        self.pheromone = np.clip(self.pheromone, self.min_pheromone, self.max_pheromone)
    
    def plot_pheromone_network(
        self,
        node_coordinates: List[Tuple[float, float]],
        routes: List[List[int]] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Візуалізує мережу з феромонами і маршрутами.
        
        Args:
            node_coordinates: Координати вузлів
            routes: Список маршрутів для відображення
            figsize: Розмір фігури
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=figsize)
        
        n_nodes = len(node_coordinates)
        
        # Нормалізуємо феромони для відображення
        max_pheromone = np.max(self.pheromone)
        normalized_pheromone = self.pheromone / max_pheromone if max_pheromone > 0 else self.pheromone
        
        # Відображаємо ребра з феромонами
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if self.pheromone[i, j] > self.min_pheromone:
                    x1, y1 = node_coordinates[i]
                    x2, y2 = node_coordinates[j]
                    
                    # Товщина лінії залежить від кількості феромону
                    line_width = 1 + 5 * normalized_pheromone[i, j]
                    
                    # Колір лінії залежить від кількості феромону
                    alpha = 0.3 + 0.7 * normalized_pheromone[i, j]
                    
                    plt.plot([x1, x2], [y1, y2], 'b-', alpha=alpha, linewidth=line_width)
        
        # Відображаємо вузли
        x_coords, y_coords = zip(*node_coordinates)
        plt.scatter(x_coords, y_coords, s=100, c='blue', edgecolors='black', zorder=10)
        
        # Відображаємо маршрути, якщо вони задані
        if routes:
            colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink']
            for i, route in enumerate(routes):
                route_x = [node_coordinates[node][0] for node in route]
                route_y = [node_coordinates[node][1] for node in route]
                
                color = colors[i % len(colors)]
                
                plt.plot(route_x, route_y, 'o-', color=color, linewidth=2.5, markersize=8, alpha=0.7, zorder=5)
        
        # Підписуємо вузли
        for i, (x, y) in enumerate(node_coordinates):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12)
        
        plt.title('Pheromone Network and Routes')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()