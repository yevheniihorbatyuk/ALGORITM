import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import time
from collections import defaultdict

class LogisticsMetrics:
    """Клас для обчислення та відображення метрик логістичної системи."""
    
    def __init__(self, network: 'LogisticsNetwork'):
        """
        Ініціалізує об'єкт оцінки метрик.
        
        Args:
            network: Об'єкт логістичної мережі
        """
        self.network = network
        self.metrics_history = defaultdict(list)
    
    def calculate_route_metrics(self, route: List[str], vehicle_id: str = None) -> Dict[str, float]:
        """
        Обчислює метрики для окремого маршруту.
        
        Args:
            route: Список ID вузлів маршруту
            vehicle_id: ID транспортного засобу (якщо None, не враховує обмеження транспорту)
            
        Returns:
            Dict[str, float]: Словник метрик
        """
        metrics = {}
        
        # Загальна відстань
        total_distance = 0
        for i in range(len(route) - 1):
            distance = self.network.get_distance(route[i], route[i+1])
            total_distance += distance
        
        metrics["total_distance"] = total_distance
        
        # Кількість відвіданих вузлів
        metrics["num_stops"] = len(route) - 2  # Без початкового і кінцевого
        
        # Середня відстань між зупинками
        metrics["avg_distance_between_stops"] = total_distance / max(1, len(route) - 1)
        
        # Якщо вказано транспортний засіб, враховуємо його характеристики
        if vehicle_id and vehicle_id in self.network.vehicles:
            vehicle = self.network.vehicles[vehicle_id]
            
            # Час в дорозі
            metrics["travel_time"] = total_distance / vehicle.avg_speed
            
            # Вартість маршруту
            metrics["cost"] = total_distance * vehicle.cost_per_km
            
            # Перевіряємо обмеження центру міста
            has_city_center_restriction = False
            if vehicle.restrictions and "city_center" in vehicle.restrictions:
                has_city_center_restriction = True
            
            # Перевіряємо, чи маршрут проходить через центр міста
            city_center_violations = 0
            for node_id in route:
                node = self.network.nodes.get(node_id)
                if node and node.node_type == NodeType.OFFICE and has_city_center_restriction:
                    city_center_violations += 1
            
            metrics["city_center_violations"] = city_center_violations
        
        return metrics
    
    def calculate_package_distribution_metrics(self, assignments: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Обчислює метрики розподілу посилок між транспортними засобами.
        
        Args:
            assignments: Словник розподілу (ключ - ID транспортного засобу, значення - список ID посилок)
            
        Returns:
            Dict[str, Any]: Словник метрик
        """
        metrics = {}
        
        # Загальна кількість посилок
        total_packages = sum(len(packages) for packages in assignments.values())
        metrics["total_packages"] = total_packages
        
        # Кількість використаних транспортних засобів
        used_vehicles = sum(1 for packages in assignments.values() if packages)
        metrics["used_vehicles"] = used_vehicles
        
        # Середня кількість посилок на транспортний засіб
        metrics["avg_packages_per_vehicle"] = total_packages / max(1, used_vehicles)
        
        # Обчислюємо завантаження кожного транспортного засобу
        vehicle_loads = {}
        for vehicle_id, package_ids in assignments.items():
            if vehicle_id not in self.network.vehicles:
                continue
                
            vehicle = self.network.vehicles[vehicle_id]
            
            # Обчислюємо сумарну вагу і об'єм
            total_weight = sum(self.network.packages[p_id].weight for p_id in package_ids if p_id in self.network.packages)
            total_volume = sum(self.network.packages[p_id].volume for p_id in package_ids if p_id in self.network.packages)
            
            # Обчислюємо коефіцієнти завантаження
            weight_ratio = total_weight / vehicle.max_weight if vehicle.max_weight > 0 else 0
            volume_ratio = total_volume / vehicle.max_volume if vehicle.max_volume > 0 else 0
            
            vehicle_loads[vehicle_id] = {
                "weight": total_weight,
                "volume": total_volume,
                "weight_ratio": weight_ratio,
                "volume_ratio": volume_ratio,
                "load_ratio": max(weight_ratio, volume_ratio),
                "num_packages": len(package_ids)
            }
        
        metrics["vehicle_loads"] = vehicle_loads
        
        # Обчислюємо загальні метрики завантаження
        if vehicle_loads:
            metrics["avg_weight_ratio"] = np.mean([load["weight_ratio"] for load in vehicle_loads.values()])
            metrics["avg_volume_ratio"] = np.mean([load["volume_ratio"] for load in vehicle_loads.values()])
            metrics["avg_load_ratio"] = np.mean([load["load_ratio"] for load in vehicle_loads.values()])
            metrics["max_load_ratio"] = max([load["load_ratio"] for load in vehicle_loads.values()])
            
            # Стандартне відхилення коефіцієнтів завантаження (для оцінки балансування)
            metrics["load_ratio_std"] = np.std([load["load_ratio"] for load in vehicle_loads.values()])
        
        return metrics


    def plot_route_map(
        self,
        routes: Dict[str, List[str]],
        figsize: Tuple[int, int] = (12, 10),
        title: str = "Карта маршрутів",
        show_node_labels: bool = True,
        colormap: str = "tab10"
    ):
        """
        Візуалізує маршрути на карті.
        
        Args:
            routes: Словник маршрутів (ключ - назва алгоритму, значення - маршрут)
            figsize: Розмір фігури
            title: Заголовок графіка
            show_node_labels: Чи відображати мітки вузлів
            colormap: Назва кольорової карти для маршрутів
        """
        plt.figure(figsize=figsize)
        
        # Отримуємо координати всіх вузлів
        node_coords = {node_id: node.location for node_id, node in self.network.nodes.items()}
        
        # Відображаємо всі вузли сірим кольором
        x_coords = [loc[0] for loc in node_coords.values()]
        y_coords = [loc[1] for loc in node_coords.values()]
        plt.scatter(x_coords, y_coords, s=80, c='lightgray', edgecolors='gray', zorder=5)
        
        # Відображаємо мітки вузлів
        if show_node_labels:
            for node_id, (x, y) in node_coords.items():
                node = self.network.nodes[node_id]
                label = f"{node_id}\n({node.node_type.value})"
                plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Кольори для маршрутів
        cmap = plt.cm.get_cmap(colormap, len(routes))
        
        # Відображаємо кожен маршрут
        for i, (name, route) in enumerate(routes.items()):
            color = cmap(i)
            
            # Отримуємо координати вузлів на маршруті
            route_x = [node_coords[node_id][0] for node_id in route if node_id in node_coords]
            route_y = [node_coords[node_id][1] for node_id in route if node_id in node_coords]
            
            # Відображаємо маршрут
            plt.plot(route_x, route_y, 'o-', color=color, linewidth=2, markersize=8, zorder=10,
                    label=f"{name} ({len(route)-1} зупинок)")
            
            # Виділяємо початковий і кінцевий вузли
            plt.scatter([route_x[0]], [route_y[0]], s=150, c='green', edgecolors='black', zorder=15)
            plt.scatter([route_x[-1]], [route_y[-1]], s=150, c='red', edgecolors='black', zorder=15)
            
            # Додаємо стрілки для позначення напрямку
            for j in range(len(route_x) - 1):
                x1, y1 = route_x[j], route_y[j]
                x2, y2 = route_x[j + 1], route_y[j + 1]
                
                # Обчислюємо середину для розміщення стрілки
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # Обчислюємо напрямок
                dx = x2 - x1
                dy = y2 - y1
                
                # Додаємо стрілку
                plt.arrow(mid_x - dx * 0.1, mid_y - dy * 0.1, dx * 0.2, dy * 0.2,
                        head_width=0.01, head_length=0.02, fc=color, ec=color, zorder=7)
        
        plt.title(title)
        plt.xlabel("Широта")
        plt.ylabel("Довгота")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()

    def plot_algorithm_convergence(
        self,
        histories: Dict[str, List[Tuple[int, float]]],
        figsize: Tuple[int, int] = (12, 6),
        title: str = "Конвергенція алгоритмів",
        is_maximizing: bool = False
    ):
        """
        Візуалізує конвергенцію різних алгоритмів.
        
        Args:
            histories: Словник історій (ключ - назва алгоритму, значення - історія значень цільової функції)
            figsize: Розмір фігури
            title: Заголовок графіка
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
        """
        plt.figure(figsize=figsize)
        
        # Відображаємо історію для кожного алгоритму
        for algorithm, history in histories.items():
            iterations = [point[0] for point in history]
            values = [point[1] for point in history]
            plt.plot(iterations, values, label=algorithm)
        
        plt.title(title)
        plt.xlabel("Ітерація")
        plt.ylabel("Значення цільової функції")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Додаємо стрілку, яка показує напрямок оптимізації
        plt.annotate(
            "Краще" if is_maximizing else "Краще",
            xy=(0.98, 0.98 if is_maximizing else 0.02),
            xycoords='axes fraction',
            xytext=(0.98, 0.9 if is_maximizing else 0.1),
            textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            horizontalalignment='right',
            verticalalignment='top' if is_maximizing else 'bottom',
        )
        
        plt.tight_layout()
        
        return plt.gcf()

    def plot_benchmark_results(
        self,
        benchmark_results: Dict[str, Dict[str, Any]],
        figsize: Tuple[int, int] = (14, 8),
        title: str = "Порівняння алгоритмів оптимізації"
    ):
        """
        Візуалізує результати бенчмарку алгоритмів.
        
        Args:
            benchmark_results: Словник результатів бенчмарку
            figsize: Розмір фігури
            title: Заголовок графіка
        """
        plt.figure(figsize=figsize)
        
        # Створюємо DataFrame для візуалізації
        data = []
        for algorithm, result in benchmark_results.items():
            data.append({
                "algorithm": algorithm,
                "avg_value": result["avg_value"],
                "avg_time": result["avg_time"],
                "avg_iterations": result["avg_iterations"],
                "std_value": result["std_value"],
                "std_time": result["std_time"],
                "std_iterations": result["std_iterations"]
            })
        
        df = pd.DataFrame(data)
        
        # Створюємо графік з двома підграфіками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Відображаємо середнє значення цільової функції
        ax1.bar(df["algorithm"], df["avg_value"], yerr=df["std_value"], capsize=5)
        ax1.set_title("Середнє значення цільової функції")
        ax1.set_ylabel("Значення")
        ax1.grid(axis="y", alpha=0.3)
        
        # Відображаємо середній час виконання
        ax2.bar(df["algorithm"], df["avg_time"], yerr=df["std_time"], capsize=5, color="orange")
        ax2.set_title("Середній час виконання")
        ax2.set_ylabel("Час (с)")
        ax2.grid(axis="y", alpha=0.3)
        
        # Додаємо значення середньої кількості ітерацій
        for i, row in df.iterrows():
            ax2.annotate(
                f"{row['avg_iterations']:.0f}",
                xy=(i, row["avg_time"]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom"
            )
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return plt.gcf()