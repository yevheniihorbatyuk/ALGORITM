import mmh3  # Для хешування
import math

class HyperLogLog:
    def __init__(self, b):
        self.b = b  # Кількість бітів для індексації
        self.m = 1 << b  # Кількість регістрів (2^b)
        self.registers = [0] * self.m

    def add(self, item):
        hash_value = mmh3.hash(item, signed=False)  # Хешування елемента
        index = hash_value >> (32 - self.b)  # Перші b бітів
        w = hash_value & ((1 << (32 - self.b)) - 1)  # Решта бітів
        leading_zeros = self._count_leading_zeros(w) + 1
        self.registers[index] = max(self.registers[index], leading_zeros)

    def _count_leading_zeros(self, w):
        return len(bin(w)) - len(bin(w).lstrip('0b'))

    def estimate(self):
        harmonic_mean = sum(2 ** -r for r in self.registers)
        raw_estimate = (self.m ** 2) / harmonic_mean
        alpha_m = 0.7213 / (1 + 1.079 / self.m)  # Коригувальний коефіцієнт
        return alpha_m * raw_estimate

if __name__ == "__main__":
    # Тестування
    hll = HyperLogLog(10)  # Використовуємо 2^10 регістрів
    elements = [f"user_{i}" for i in range(1, 10000)]

    for elem in elements:
        hll.add(elem)

    print("Оцінка кількості унікальних елементів:", round(hll.estimate()))