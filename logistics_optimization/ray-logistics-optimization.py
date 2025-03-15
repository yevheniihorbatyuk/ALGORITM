import ray
import time
import random
import numpy as np
from copy import deepcopy
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Ініціалізація Ray
ray.init(address="auto", ignore_reinit_error=True)  # Підключення до існуючого кластера або створення локального


@dataclass
class Point:
    """Клас для представлення точки на карті."""
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        """Обчислює відстань до іншої точки."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class Warehouse:
    """Клас для представлення складу."""
    id: int
    location: Point


@dataclass
class Order:
    """Клас для представлення замовлення."""
    id: int
    location: Point
    volume: float  # об'єм замовлення
    warehouse_id: int  # ID складу, з якого замовлення повинно бути доставлене


@dataclass
class Truck:
    """Клас для представлення вантажівки."""
    id: int
    capacity: float  # вантажопідйомність
    current_load: float = 0.0
    home_warehouse_id: int = 0  # склад, до якого приписана вантажівка

    def __str__(self) -> str:
        return f"Truck {self.id} (Load: {self.current_load:.1f}/{self.capacity:.1f})"


@dataclass
class Problem:
    """Клас для представлення задачі оптимізації."""
    warehouses: List[Warehouse]
    orders: List[Order]
    trucks: List[Truck]


class Route:
    """Клас для представлення маршруту."""
    def __init__(self, truck: Truck, problem: Problem):
        self.truck = truck
        self.orders: List[Order] = []
        self.problem = problem

    def compute_distance(self) -> float:
        """Обчислює загальну відстань маршруту."""
        if not self.orders:
            return 0.0

        # Знаходимо домашній склад
        home_warehouse = next(w for w in self.problem.warehouses if w.id == self.truck.home_warehouse_id)
        total_distance = 0.0
        
        # Від складу до першого замовлення
        total_distance += home_warehouse.location.distance_to(self.orders[0].location)
        
        # Між замовленнями
        for i in range(len(self.orders) - 1):
            total_distance += self.orders[i].location.distance_to(self.orders[i + 1].location)
        
        # Від останнього замовлення назад до складу
        if self.orders:
            total_distance += self.orders[-1].location.distance_to(home_warehouse.location)
            
        return total_distance

    def __str__(self) -> str:
        route_str = f"Route for {self.truck}, Orders: {len(self.orders)}, "
        route_str += f"Distance: {self.compute_distance():.1f}"
        return route_str


class Solution:
    """Клас для представлення розв'язку задачі."""
    def __init__(self, problem: Problem):
        self.problem = problem
        self.routes: List[Route] = []

    def compute_total_cost(self) -> float:
        """Обчислює загальну вартість розв'язку."""
        return sum(route.compute_distance() for route in self.routes)

    def is_valid(self) -> bool:
        """Перевіряє, чи є розв'язок валідним."""
        # Перевіряємо чи всі замовлення розподілені
        all_orders_in_solution = []
        for route in self.routes:
            all_orders_in_solution.extend([order.id for order in route.orders])
            
        all_orders_in_problem = [order.id for order in self.problem.orders]
        
        if set(all_orders_in_solution) != set(all_orders_in_problem):
            return False
            
        # Перевіряємо чи вантажопідйомність не перевищена
        for route in self.routes:
            total_volume = sum(order.volume for order in route.orders)
            if total_volume > route.truck.capacity:
                return False
                
        return True


@ray.remote
class LocalSearchOperator:
    """Клас для операцій локального пошуку, адаптований для Ray."""

    @staticmethod
    def swap_orders(solution: Solution, route1_idx: int, order1_idx: int,
                   route2_idx: int, order2_idx: int) -> bool:
        """
        Обмінює два замовлення між маршрутами.
        Повертає True, якщо обмін можливий і був виконаний.
        """
        if route1_idx == route2_idx:
            return False

        route1 = solution.routes[route1_idx]
        route2 = solution.routes[route2_idx]

        if order1_idx >= len(route1.orders) or order2_idx >= len(route2.orders):
            return False

        order1 = route1.orders[order1_idx]
        order2 = route2.orders[order2_idx]

        # Перевіряємо можливість обміну (вантажопідйомність)
        new_load1 = route1.truck.current_load - order1.volume + order2.volume
        new_load2 = route2.truck.current_load - order2.volume + order1.volume

        if (new_load1 <= route1.truck.capacity and
            new_load2 <= route2.truck.capacity):
            # Виконуємо обмін
            route1.orders[order1_idx] = order2
            route2.orders[order2_idx] = order1

            # Оновлюємо завантаження
            route1.truck.current_load = new_load1
            route2.truck.current_load = new_load2
            return True

        return False

    @staticmethod
    def relocate_order(solution: Solution, from_route_idx: int,
                      order_idx: int, to_route_idx: int,
                      new_position: int) -> bool:
        """
        Переміщує замовлення з одного маршруту в інший.
        """
        if from_route_idx == to_route_idx:
            return False

        from_route = solution.routes[from_route_idx]
        to_route = solution.routes[to_route_idx]

        if order_idx >= len(from_route.orders):
            return False

        order = from_route.orders[order_idx]

        # Перевіряємо можливість переміщення
        new_load = to_route.truck.current_load + order.volume
        if new_load <= to_route.truck.capacity:
            # Видаляємо з початкового маршруту
            from_route.orders.pop(order_idx)
            from_route.truck.current_load -= order.volume

            # Додаємо до нового маршруту
            to_route.orders.insert(min(new_position, len(to_route.orders)), order)
            to_route.truck.current_load += order.volume
            return True

        return False

    @staticmethod
    def two_opt_swap(solution: Solution, route_idx: int, i: int, j: int) -> bool:
        """
        Виконує 2-opt swap в межах одного маршруту (перевертає частину маршруту).
        """
        route = solution.routes[route_idx]
        if i >= j or i < 0 or j >= len(route.orders):
            return False

        # Перевертаємо частину маршруту між i та j
        route.orders[i:j+1] = reversed(route.orders[i:j+1])
        return True


@ray.remote
def generate_neighbor(solution: Solution, operation_type: Optional[str] = None) -> Tuple[Solution, str]:
    """
    Генерує випадкового сусіда поточного розв'язку.
    Повертає (новий_розв'язок, тип_операції).
    """
    new_solution = deepcopy(solution)

    if operation_type is None:
        operation = random.choice(['swap', 'relocate', 'two_opt'])
    else:
        operation = operation_type

    if operation == 'swap':
        if len(new_solution.routes) >= 2:
            route_indices = random.sample(range(len(new_solution.routes)), 2)
            route1_idx, route2_idx = route_indices[0], route_indices[1]

            route1 = new_solution.routes[route1_idx]
            route2 = new_solution.routes[route2_idx]

            if route1.orders and route2.orders:
                order1_idx = random.randint(0, len(route1.orders) - 1)
                order2_idx = random.randint(0, len(route2.orders) - 1)

                if LocalSearchOperator.swap_orders(new_solution, route1_idx, order1_idx, route2_idx, order2_idx):
                    return new_solution, 'swap'

    elif operation == 'relocate':
        if len(new_solution.routes) >= 2:
            route_indices = random.sample(range(len(new_solution.routes)), 2)
            from_route_idx, to_route_idx = route_indices[0], route_indices[1]

            from_route = new_solution.routes[from_route_idx]

            if from_route.orders:
                order_idx = random.randint(0, len(from_route.orders) - 1)
                new_position = random.randint(0, len(new_solution.routes[to_route_idx].orders))

                if LocalSearchOperator.relocate_order(new_solution, from_route_idx, order_idx, to_route_idx, new_position):
                    return new_solution, 'relocate'

    elif operation == 'two_opt':
        # Оптимізовано: пробуємо 2-opt для випадкового маршруту з достатньою кількістю замовлень
        eligible_routes = [idx for idx, route in enumerate(new_solution.routes) if len(route.orders) >= 4]
        
        if eligible_routes:
            for _ in range(3):  # Пробуємо до 3 разів
                route_idx = random.choice(eligible_routes)
                route = new_solution.routes[route_idx]
                num_orders = len(route.orders)
                
                i = random.randint(0, num_orders - 3)  # Гарантуємо мінімум 2 замовлення для перевороту
                j = random.randint(i + 1, num_orders - 1)

                if LocalSearchOperator.two_opt_swap(new_solution, route_idx, i, j):
                    return new_solution, 'two_opt'

    # Якщо жодна операція не вдалася, повертаємо копію початкового розв'язку
    return new_solution, 'none'


@ray.remote
def evaluate_solution(solution: Solution) -> float:
    """Обчислює вартість розв'язку."""
    return solution.compute_total_cost()


@ray.remote
class WorkerState:
    """Клас для зберігання стану роботи кожного процесу."""
    def __init__(self, solution: Solution, worker_id: int):
        self.current_solution = solution
        self.best_solution = deepcopy(solution)
        self.best_cost = solution.compute_total_cost()
        self.iterations = 0
        self.no_improve = 0
        self.worker_id = worker_id

    def get_state(self) -> Dict:
        """Повертає поточний стан."""
        return {
            "worker_id": self.worker_id,
            "iterations": self.iterations,
            "no_improve": self.no_improve,
            "best_cost": self.best_cost
        }
    
    def get_best_solution(self) -> Tuple[Solution, float]:
        """Повертає найкраще знайдене рішення та його вартість."""
        return self.best_solution, self.best_cost

    def update(self, neighbor_solution: Solution, neighbor_cost: float) -> bool:
        """Оновлює стан на основі нового рішення."""
        self.iterations += 1
        
        if neighbor_cost < self.best_cost:
            self.best_solution = deepcopy(neighbor_solution)
            self.best_cost = neighbor_cost
            self.current_solution = neighbor_solution
            self.no_improve = 0
            return True
        else:
            self.no_improve += 1
            return False


def parallel_hill_climbing(initial_solution: Solution,
                          max_iterations: int = 1000,
                          max_no_improve: int = 100,
                          num_workers: int = 4) -> Solution:
    """
    Реалізує паралельний алгоритм Hill Climbing для покращення розв'язку.

    Args:
        initial_solution: початковий розв'язок
        max_iterations: максимальна кількість ітерацій
        max_no_improve: максимальна кількість ітерацій без покращення
        num_workers: кількість паралельних процесів
    """
    start_time = time.time()
    
    # Створюємо стан для кожного процесу
    worker_states = [WorkerState.remote(deepcopy(initial_solution), i) for i in range(num_workers)]
    
    # Основний цикл оптимізації
    active_workers = list(range(num_workers))
    all_improvements = 0
    
    # Підготовка різних типів операцій для диверсифікації пошуку
    operation_types = ['swap', 'relocate', 'two_opt', None]  # None означає випадковий вибір
    
    while active_workers:
        # Отримуємо поточний стан процесів
        states_futures = [worker_states[i].get_state.remote() for i in active_workers]
        states = ray.get(states_futures)
        
        # Запускаємо генерацію сусідів для активних процесів
        futures = []
        for i, worker_idx in enumerate(active_workers):
            state = states[i]
            worker = worker_states[worker_idx]
            
            # Якщо досягнуто умов завершення, видалити процес зі списку активних
            if state["iterations"] >= max_iterations or state["no_improve"] >= max_no_improve:
                active_workers.remove(worker_idx)
                continue
                
            # Отримуємо поточне рішення для процесу
            current_solution_future = worker.get_best_solution.remote()
            current_solution, _ = ray.get(current_solution_future)
            
            # Вибираємо тип операції з урахуванням диверсифікації для різних процесів
            operation_type = operation_types[worker_idx % len(operation_types)]
            
            # Генеруємо кілька сусідів для кожного процесу
            num_neighbors = 5  # Кількість сусідів для генерації
            neighbor_futures = [generate_neighbor.remote(current_solution, operation_type) for _ in range(num_neighbors)]
            
            # Оцінка всіх сусідів
            for neighbor_future in neighbor_futures:
                neighbor, operation = ray.get(neighbor_future)
                if operation != 'none':  # Перевіряємо, чи була застосована операція
                    cost_future = evaluate_solution.remote(neighbor)
                    futures.append((worker, neighbor, cost_future, worker_idx))
        
        # Обробка результатів
        for worker, neighbor, cost_future, worker_idx in futures:
            cost = ray.get(cost_future)
            improved = ray.get(worker.update.remote(neighbor, cost))
            if improved:
                all_improvements += 1
                if all_improvements % 10 == 0:  # Логування кожні 10 покращень
                    elapsed_time = time.time() - start_time
                    print(f"[{elapsed_time:.1f}s] Total improvements: {all_improvements}, "
                          f"Best cost so far: {ray.get(worker.get_state.remote())['best_cost']:.2f}")
        
        # Синхронізуємо найкращі рішення між процесами кожні N ітерацій
        if all_improvements % 50 == 0 and active_workers:
            # Отримуємо найкращі рішення від всіх процесів
            best_solutions = [ray.get(worker_states[i].get_best_solution.remote()) for i in active_workers]
            best_solution, best_cost = min(best_solutions, key=lambda x: x[1])
            
            # Оновлюємо найкраще рішення для всіх процесів
            for worker_idx in active_workers:
                worker = worker_states[worker_idx]
                ray.get(worker.update.remote(best_solution, best_cost))
    
    # Отримуємо найкращі рішення від всіх процесів
    best_solutions = [ray.get(worker_states[i].get_best_solution.remote()) for i in range(num_workers)]
    best_solution, best_cost = min(best_solutions, key=lambda x: x[1])
    
    elapsed_time = time.time() - start_time
    print(f"\nParallel Hill Climbing finished in {elapsed_time:.2f} seconds")
    print(f"Initial cost: {initial_solution.compute_total_cost():.2f}")
    print(f"Final cost: {best_cost:.2f}")
    print(f"Improvement: {(initial_solution.compute_total_cost() - best_cost):.2f} "
          f"({(initial_solution.compute_total_cost() - best_cost) / initial_solution.compute_total_cost() * 100:.1f}%)")
    
    return best_solution


def generate_test_problem(num_warehouses: int = 2,
                        num_orders: int = 50,
                        num_trucks: int = 5,
                        area_size: int = 100,
                        seed: int = 42) -> Problem:
    """
    Генерує випадкову тестову задачу.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Створюємо склади
    warehouses = []
    for i in range(num_warehouses):
        location = Point(x=random.uniform(0, area_size), y=random.uniform(0, area_size))
        warehouses.append(Warehouse(id=i, location=location))
    
    # Створюємо замовлення
    orders = []
    for i in range(num_orders):
        location = Point(x=random.uniform(0, area_size), y=random.uniform(0, area_size))
        volume = random.uniform(1, 10)  # Випадковий об'єм від 1 до 10
        warehouse_id = random.randint(0, num_warehouses - 1)  # Випадковий склад
        orders.append(Order(id=i, location=location, volume=volume, warehouse_id=warehouse_id))
    
    # Створюємо вантажівки
    trucks = []
    for i in range(num_trucks):
        capacity = random.uniform(50, 100)  # Випадкова вантажопідйомність від 50 до 100
        home_warehouse_id = random.randint(0, num_warehouses - 1)  # Випадковий домашній склад
        trucks.append(Truck(id=i, capacity=capacity, home_warehouse_id=home_warehouse_id))
    
    return Problem(warehouses=warehouses, orders=orders, trucks=trucks)


@ray.remote
def greedy_solution_for_warehouse(warehouse_id: int, problem: Problem) -> List[Route]:
    """
    Створює жадібне рішення для одного складу.
    """
    # Відфільтровуємо замовлення, що належать до цього складу
    warehouse_orders = [order for order in problem.orders if order.warehouse_id == warehouse_id]
    
    # Відфільтровуємо вантажівки, що базуються на цьому складі
    warehouse_trucks = [truck for truck in problem.trucks if truck.home_warehouse_id == warehouse_id]
    
    if not warehouse_trucks:
        return []
    
    # Створюємо маршрути для вантажівок
    routes = []
    for truck in warehouse_trucks:
        route = Route(truck=deepcopy(truck), problem=problem)
        routes.append(route)
    
    # Сортуємо замовлення за об'ємом (від більшого до меншого)
    warehouse_orders.sort(key=lambda o: o.volume, reverse=True)
    
    # Розподіляємо замовлення по вантажівках
    for order in warehouse_orders:
        # Шукаємо найкращу вантажівку для цього замовлення
        best_route = None
        best_additional_distance = float('inf')
        
        for route in routes:
            # Перевіряємо чи вантажівка може взяти це замовлення
            if route.truck.current_load + order.volume <= route.truck.capacity:
                # Знаходимо склад
                warehouse = next(w for w in problem.warehouses if w.id == warehouse_id)
                
                # Обчислюємо додаткову відстань для додавання цього замовлення
                additional_distance = 0
                
                if not route.orders:
                    # Якщо маршрут порожній, обчислюємо відстань від складу до замовлення і назад
                    additional_distance = 2 * warehouse.location.distance_to(order.location)
                else:
                    # Інакше обчислюємо відстань від останнього замовлення до нового і від нового до складу
                    last_order = route.orders[-1]
                    old_return_distance = last_order.location.distance_to(warehouse.location)
                    new_to_last_distance = last_order.location.distance_to(order.location)
                    new_return_distance = order.location.distance_to(warehouse.location)
                    additional_distance = new_to_last_distance + new_return_distance - old_return_distance
                
                if additional_distance < best_additional_distance:
                    best_additional_distance = additional_distance
                    best_route = route
        
        # Якщо знайдено підходящу вантажівку, додаємо замовлення до неї
        if best_route:
            best_route.orders.append(order)
            best_route.truck.current_load += order.volume
    
    return routes


def greedy_solution(problem: Problem) -> Solution:
    """
    Створює початкове рішення за допомогою жадібного алгоритму.
    Розпаралелює створення рішення для кожного складу.
    """
    solution = Solution(problem)
    
    # Отримуємо всі унікальні ID складів
    warehouse_ids = set(warehouse.id for warehouse in problem.warehouses)
    
    # Запускаємо паралельне обчислення для кожного складу
    futures = [greedy_solution_for_warehouse.remote(wid, problem) for wid in warehouse_ids]
    warehouse_routes = ray.get(futures)
    
    # Об'єднуємо всі маршрути
    for routes in warehouse_routes:
        solution.routes.extend(routes)
    
    # Перевіряємо чи всі замовлення розподілені
    assigned_orders = set()
    for route in solution.routes:
        for order in route.orders:
            assigned_orders.add(order.id)
    
    all_orders = set(order.id for order in problem.orders)
    unassigned_orders = all_orders - assigned_orders
    
    # Якщо є нерозподілені замовлення, створюємо для них нові маршрути
    if unassigned_orders:
        print(f"Warning: {len(unassigned_orders)} orders could not be assigned in the greedy solution.")
        
        # Спроба розподілити нерозподілені замовлення
        unassigned_order_objects = [order for order in problem.orders if order.id in unassigned_orders]
        
        # Сортуємо маршрути за вільним місцем
        routes_with_space = sorted(solution.routes, key=lambda r: r.truck.capacity - r.truck.current_load, reverse=True)
        
        for order in unassigned_order_objects:
            assigned = False
            for route in routes_with_space:
                if route.truck.current_load + order.volume <= route.truck.capacity:
                    route.orders.append(order)
                    route.truck.current_load += order.volume
                    assigned = True
                    break
            
            if not assigned:
                print(f"Could not assign order {order.id} with volume {order.volume}.")
    
    return solution


def visualize_solution(solution: Solution, title: str):
    """
    Візуалізує розв'язок задачі.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Визначаємо кольори для кожного маршруту
    colors = plt.cm.tab20(np.linspace(0, 1, len(solution.routes)))
    
    # Малюємо склади
    for warehouse in solution.problem.warehouses:
        ax.scatter(warehouse.location.x, warehouse.location.y, c='black', s=100, marker='s', label=f'Warehouse {warehouse.id}')
        ax.text(warehouse.location.x + 2, warehouse.location.y + 2, f'W{warehouse.id}', fontsize=12)
    
    # Малюємо маршрути
    for i, route in enumerate(solution.routes):
        if not route.orders:
            continue
            
        color = colors[i % len(colors)]
        
        # Знаходимо склад
        home_warehouse = next(w for w in solution.problem.warehouses if w.id == route.truck.home_warehouse_id)
        
        # Малюємо лінію від складу до першого замовлення
        first_order = route.orders[0]
        ax.plot([home_warehouse.location.x, first_order.location.x],
               [home_warehouse.location.y, first_order.location.y],
               c=color, linewidth=1, alpha=0.7)
        
        # Малюємо лінії між замовленнями
        for j in range(len(route.orders) - 1):
            order1 = route.orders[j]
            order2 = route.orders[j + 1]
            ax.plot([order1.location.x, order2.location.x],
                   [order1.location.y, order2.location.y],
                   c=color, linewidth=1, alpha=0.7)
        
        # Малюємо лінію від останнього замовлення назад до складу
        last_order = route.orders[-1]
        ax.plot([last_order.location.x, home_warehouse.location.x],
               [last_order.location.y, home_warehouse.location.y],
               c=color, linewidth=1, alpha=0.7, label=f'Truck {route.truck.id}')
        
        # Малюємо замовлення
        for order in route.orders:
            ax.scatter(order.location.x, order.location.y, c=color, s=30, alpha=0.8)
    
    # Налаштовуємо графік
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Додаємо інформацію про загальну вартість
    ax.text(0.02, 0.02, f'Total Cost: {solution.compute_total_cost():.2f}',
           transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    return fig


# Виконання задачі з використанням Ray
if __name__ == "__main__":
    # Створюємо тестову задачу
    problem = generate_test_problem(
        num_warehouses=2,
        num_orders=1000,
        num_trucks=6,
        area_size=1000,
        seed=300
    )

    # Знаходимо початковий розв'язок жадібним алгоритмом
    print("Генерація початкового розв'язку...")
    initial_solution = greedy_solution(problem)
    print(f"Початковий розв'язок, вартість: {initial_solution.compute_total_cost():.2f}")

    # Покращуємо розв'язок за допомогою паралельного Hill Climbing
    print("\nЗапуск паралельного Hill Climbing...")
    
    # Визначаємо кількість ядер CPU для паралельної обробки
    num_cpus = ray.cluster_resources().get("CPU", 4)
    num_workers = max(4, int(num_cpus * 0.8))  # Використовуємо 80% доступних ядер