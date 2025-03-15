
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np
import uuid
import pandas as pd

class NodeType(Enum):
    """Типи вузлів логістичної мережі."""
    OFFICE = "office"        # Відділення - прийом/видача
    SORTING = "sorting"      # Сортувальний центр
    LOGISTICS = "logistics"  # Логістичний центр (склад)

class PackagePriority(Enum):
    """Пріоритет посилки."""
    STANDARD = "standard"    # Стандартна доставка
    EXPRESS = "express"      # Термінова доставка
    VIP = "vip"              # VIP-доставка

class VehicleType(Enum):
    """Типи транспортних засобів."""
    SMALL = "small"          # Малі вантажівки (до 3.5т)
    MEDIUM = "medium"        # Середні вантажівки (до 7.5т)
    LARGE = "large"          # Великі вантажівки (понад 7.5т)

@dataclass
class Node:
    """Клас, що представляє вузол логістичної мережі."""
    id: str
    name: str
    node_type: NodeType
    location: Tuple[float, float]  # (широта, довгота)
    capacity: float                # Макс. обсяг обробки посилок
    working_hours: Tuple[float, float]  # (початок, кінець) у годинах
    
    @property
    def is_open(self, current_time: float) -> bool:
        """Перевіряє, чи відкрито вузол у вказаний час."""
        start, end = self.working_hours
        # Перетворюємо current_time у години доби (0-24)
        time_of_day = current_time % 24
        return start <= time_of_day <= end

@dataclass
class Depot(Node):
    """Клас, що представляє депо (базу) для транспортних засобів."""
    max_vehicles: int = 10  # Максимальна кількість транспортних засобів
    stationed_vehicles: List[str] = field(default_factory=list)  # ID транспортних засобів на базі
    
    def add_vehicle(self, vehicle_id: str):
        """Додає транспортний засіб до депо."""
        if len(self.stationed_vehicles) < self.max_vehicles:
            self.stationed_vehicles.append(vehicle_id)
            return True
        return False
        
    def remove_vehicle(self, vehicle_id: str):
        """Видаляє транспортний засіб з депо."""
        if vehicle_id in self.stationed_vehicles:
            self.stationed_vehicles.remove(vehicle_id)
            return True
        return False

@dataclass
class Package:
    """Клас, що представляє посилку в системі."""
    id: str
    weight: float            # Вага в кг
    volume: float            # Об'єм в м³
    origin_id: str           # ID вузла відправлення
    destination_id: str      # ID вузла призначення
    priority: PackagePriority
    creation_time: float     # Час створення посилки
    delivery_deadline: Optional[float] = None  # Крайній термін доставки
    pickup_time_window: Optional[Tuple[float, float]] = None  # Часове вікно для забору
    delivery_time_window: Optional[Tuple[float, float]] = None  # Часове вікно для доставки
    special_requirements: List[str] = field(default_factory=list)  # Особливі вимоги
    status: str = "created"  # Статус посилки (created, processing, in_transit, delivered)
    
    def is_deliverable_at(self, current_time: float) -> bool:
        """Перевіряє, чи можлива доставка посилки в поточний час."""
        if self.delivery_time_window is None:
            return True
        start, end = self.delivery_time_window
        return start <= current_time <= end
    
    def is_late(self, current_time: float) -> bool:
        """Перевіряє, чи прострочено доставку посилки."""
        if self.delivery_deadline is None:
            return False
        return current_time > self.delivery_deadline
    
    def get_priority_score(self, current_time: float) -> float:
        """Обчислює пріоритетний бал посилки (вищий бал = вищий пріоритет)."""
        # Базовий бал залежно від пріоритету
        if self.priority == PackagePriority.VIP:
            base_score = 10.0
        elif self.priority == PackagePriority.EXPRESS:
            base_score = 5.0
        else:  # STANDARD
            base_score = 1.0
            
        # Збільшуємо бал, якщо наближається дедлайн
        if self.delivery_deadline is not None:
            time_left = self.delivery_deadline - current_time
            if time_left > 0:
                # Чим менше часу залишилось, тим вищий бал
                urgency_factor = max(1.0, 10.0 / max(1.0, time_left))
                base_score *= urgency_factor
                
        return base_score

@dataclass
class Vehicle:
    """Клас, що представляє транспортний засіб."""
    id: str
    vehicle_type: VehicleType
    max_weight: float        # Максимальна вантажопідйомність у кг
    max_volume: float        # Максимальний об'єм у м³
    avg_speed: float         # Середня швидкість км/год
    cost_per_km: float       # Вартість експлуатації за км
    current_location: str    # ID поточного вузла
    home_depot_id: str       # ID домашнього депо
    max_working_hours: float = 8.0  # Максимальний робочий час водія в годинах
    restrictions: List[str] = field(default_factory=list)  # Обмеження (центр міста тощо)
    current_route: Optional[str] = None  # ID поточного маршруту
    current_load_weight: float = 0.0     # Поточне завантаження за вагою
    current_load_volume: float = 0.0     # Поточне завантаження за об'ємом
    
    def has_capacity_for(self, package: Package) -> bool:
        """Перевіряє, чи є у транспорту достатня ємність для посилки."""
        return (self.current_load_weight + package.weight <= self.max_weight and
                self.current_load_volume + package.volume <= self.max_volume)
    
    def add_package(self, package: Package) -> bool:
        """Додає посилку до транспорту."""
        if self.has_capacity_for(package):
            self.current_load_weight += package.weight
            self.current_load_volume += package.volume
            return True
        return False
    
    def get_remaining_capacity(self) -> Tuple[float, float]:
        """Повертає залишкову ємність транспорту (вага, об'єм)."""
        return (self.max_weight - self.current_load_weight, 
                self.max_volume - self.current_load_volume)

@dataclass
class Node:
    """Клас, що представляє вузол логістичної мережі."""
    id: str
    name: str
    node_type: NodeType
    location: Tuple[float, float]  # (широта, довгота)
    capacity: float                # Макс. обсяг обробки посилок
    working_hours: Tuple[float, float]  # (початок, кінець) у годинах
    
    @property
    def is_open(self, current_time: float) -> bool:
        """Перевіряє, чи відкрито вузол у вказаний час."""
        start, end = self.working_hours
        # Перетворюємо current_time у години доби (0-24)
        time_of_day = current_time % 24
        return start <= time_of_day <= end

@dataclass
class Route:
    """Клас, що представляє маршрут."""
    id: str
    vehicle_id: str
    nodes: List[str]         # Послідовність ID вузлів
    packages: List[str]      # ID посилок на маршруті
    departure_time: float    # Час відправлення
    estimated_arrival: Dict[str, float]  # Очікуваний час прибуття до кожного вузла
    total_distance: float    # Загальна відстань у км
    total_time: float        # Загальний час у годинах
    
    @property
    def load_weight(self, packages_dict: Dict[str, Package]) -> float:
        """Розраховує загальну вагу посилок на маршруті."""
        return sum(packages_dict[p_id].weight for p_id in self.packages)
    
    @property
    def load_volume(self, packages_dict: Dict[str, Package]) -> float:
        """Розраховує загальний об'єм посилок на маршруті."""
        return sum(packages_dict[p_id].volume for p_id in self.packages)

class LogisticsNetwork:
    """Клас, що представляє логістичну мережу."""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.packages: Dict[str, Package] = {}
        self.routes: Dict[str, Route] = {}
        self.distances: Dict[Tuple[str, str], float] = {}  # Відстані між вузлами
        self.travel_times: Dict[Tuple[str, str], float] = {}  # Часи подорожі між вузлами
        self.current_time: float = 0.0  # Поточний час в годинах
        
    def add_depot(self, depot: Depot):
        """Додає депо до мережі."""
        self.nodes[depot.id] = depot
        
    def assign_vehicle_to_depot(self, vehicle_id: str, depot_id: str) -> bool:
        """Призначає транспортний засіб до депо."""
        if vehicle_id not in self.vehicles or depot_id not in self.nodes:
            return False
            
        depot = self.nodes[depot_id]
        if not isinstance(depot, Depot):
            return False
            
        vehicle = self.vehicles[vehicle_id]
        vehicle.home_depot_id = depot_id
        vehicle.current_location = depot_id
        
        return depot.add_vehicle(vehicle_id)
    
    def get_travel_time(self, node1_id: str, node2_id: str, vehicle_id: Optional[str] = None) -> float:
        """
        Повертає час подорожі між двома вузлами.
        
        Args:
            node1_id: ID першого вузла
            node2_id: ID другого вузла
            vehicle_id: ID транспортного засобу (для врахування швидкості)
            
        Returns:
            float: Час подорожі в годинах
        """
        distance = self.get_distance(node1_id, node2_id)
        
        if distance == float('inf'):
            return float('inf')
            
        if vehicle_id is not None and vehicle_id in self.vehicles:
            # Використовуємо швидкість конкретного транспортного засобу
            vehicle = self.vehicles[vehicle_id]
            return distance / vehicle.avg_speed
        else:
            # Використовуємо базовий час подорожі
            return self.travel_times.get((node1_id, node2_id), distance / 50.0)  # За замовчуванням 50 км/год
    
    def get_route_time(self, route: List[str], vehicle_id: Optional[str] = None) -> float:
        """
        Обчислює загальний час проходження маршруту.
        
        Args:
            route: Список ID вузлів у маршруті
            vehicle_id: ID транспортного засобу
            
        Returns:
            float: Загальний час у годинах
        """
        total_time = 0.0
        
        for i in range(len(route) - 1):
            total_time += self.get_travel_time(route[i], route[i+1], vehicle_id)
            
        return total_time
    
    def create_route(self, vehicle_id: str, node_sequence: List[str], packages: List[str]) -> Optional[str]:
        """
        Створює новий маршрут для транспортного засобу.
        
        Args:
            vehicle_id: ID транспортного засобу
            node_sequence: Послідовність ID вузлів
            packages: Список ID посилок
            
        Returns:
            Optional[str]: ID створеного маршруту або None у разі помилки
        """
        if vehicle_id not in self.vehicles:
            return None
            
        # Перевіряємо, чи всі вузли існують
        for node_id in node_sequence:
            if node_id not in self.nodes:
                return None
                
        # Перевіряємо, чи всі посилки існують
        for package_id in packages:
            if package_id not in self.packages:
                return None
        
        vehicle = self.vehicles[vehicle_id]
        
        # Перевіряємо обмеження вантажопідйомності
        total_weight = sum(self.packages[p_id].weight for p_id in packages)
        total_volume = sum(self.packages[p_id].volume for p_id in packages)
        
        if total_weight > vehicle.max_weight or total_volume > vehicle.max_volume:
            return None
            
        # Обчислюємо загальну відстань і час
        total_distance = 0.0
        for i in range(len(node_sequence) - 1):
            total_distance += self.get_distance(node_sequence[i], node_sequence[i+1])
            
        total_time = self.get_route_time(node_sequence, vehicle_id)
        
        # Перевіряємо обмеження робочого часу
        if total_time > vehicle.max_working_hours:
            return None
            
        # Створюємо маршрут
        route_id = f"route_{len(self.routes) + 1}"
        
        estimated_arrival = {}
        current_time = self.current_time
        
        for i, node_id in enumerate(node_sequence):
            if i > 0:
                travel_time = self.get_travel_time(node_sequence[i-1], node_id, vehicle_id)
                current_time += travel_time
            estimated_arrival[node_id] = current_time
            
        route = Route(
            id=route_id,
            vehicle_id=vehicle_id,
            nodes=node_sequence,
            packages=packages,
            departure_time=self.current_time,
            estimated_arrival=estimated_arrival,
            total_distance=total_distance,
            total_time=total_time
        )
        
        self.routes[route_id] = route
        vehicle.current_route = route_id
        
        return route_id