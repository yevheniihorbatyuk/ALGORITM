import random
import time
import pandas as pd
import numpy as np
from functools import lru_cache
from collections import defaultdict

# Ініціалізація параметрів
N = 100000  # Розмір масиву
Q = 50000   # Кількість запитів
array = [random.randint(1, 100) for _ in range(N)]  # Генерація масиву

# Створюємо NumPy array для швидших операцій
np_array = np.array(array)

# Генерація запитів (50% - сума відрізку, 50% - оновлення)
queries = [
    ("Range", L := random.randint(0, N - 1), random.randint(L, N - 1))
    if random.random() < 0.5
    else ("Update", random.randint(0, N - 1), random.randint(1, 100))
    for _ in range(Q)
]


### 🔹 1. Без кешування (базовий метод)
def range_sum_no_cache(L, R):
    return sum(array[L:R + 1])

def update_no_cache(index, value):
    array[index] = value

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_no_cache(query[1], query[2])
    elif query[0] == "Update":
        update_no_cache(query[1], query[2])
no_cache_time = time.time() - start_time


### 🔹 2. Використання NumPy для прискорення
def range_sum_numpy(L, R):
    return np.sum(np_array[L:R + 1])

def update_numpy(index, value):
    np_array[index] = value

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_numpy(query[1], query[2])
    elif query[0] == "Update":
        update_numpy(query[1], query[2])
numpy_time = time.time() - start_time


### 🔹 3. Префіксні суми (без кешування)
# Створюємо масив префіксних сум
prefix_sum_array = [0] * (N + 1)
for i in range(N):
    prefix_sum_array[i + 1] = prefix_sum_array[i] + array[i]

def range_sum_prefix(L, R):
    return prefix_sum_array[R + 1] - prefix_sum_array[L]

def update_prefix(index, value):
    delta = value - array[index]
    array[index] = value
    for i in range(index + 1, N + 1):
        prefix_sum_array[i] += delta

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_prefix(query[1], query[2])
    elif query[0] == "Update":
        update_prefix(query[1], query[2])
prefix_sum_time = time.time() - start_time


### 🔹 4. Кеш з оптимізованим відстеженням залежностей
index_to_ranges = defaultdict(set)

@lru_cache(maxsize=10000)
def range_sum_optimized_cache(L, R):
    # Реєструємо залежність: кожен індекс у діапазоні [L, R] впливає на цей кеш
    for i in range(L, R + 1):
        index_to_ranges[i].add((L, R))
    return sum(array[L:R + 1])

def update_optimized_cache(index, value):
    array[index] = value
    # Знаходимо тільки ті діапазони, на які впливає цей індекс
    affected_ranges = index_to_ranges[index].copy()
    # Очищаємо кеш тільки для цих діапазонів
    for L, R in affected_ranges:
        try:
            range_sum_optimized_cache.cache_clear()  # Тут краще було б очистити конкретний ключ
            # Видаляємо діапазон з усіх індексів, які він містить
            for i in range(L, R + 1):
                index_to_ranges[i].discard((L, R))
        except:
            pass

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_optimized_cache(query[1], query[2])
    elif query[0] == "Update":
        update_optimized_cache(query[1], query[2])
optimized_cache_time = time.time() - start_time


### 🔹 5. Кеш з лічильником інвалідації
invalidation_counter = 0

@lru_cache(maxsize=10000)
def range_sum_counter_cache(L, R, counter):
    # Ігноруємо counter в обчисленнях, він потрібен лише для інвалідації кешу
    return sum(array[L:R + 1])

def update_counter_cache(index, value):
    global invalidation_counter
    array[index] = value
    # Просто збільшуємо лічильник при кожному оновленні
    invalidation_counter += 1

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_counter_cache(query[1], query[2], invalidation_counter)
    elif query[0] == "Update":
        update_counter_cache(query[1], query[2])
counter_cache_time = time.time() - start_time


### 🔹 6. Блочне кешування (розділяємо масив на блоки)
block_size = int(N**0.5)  # Оптимальний розмір блоку
block_sums = [sum(array[i:i + block_size]) for i in range(0, N, block_size)]

def range_sum_block_cache(L, R):
    result = 0
    # Обробляємо початковий неповний блок
    start_block = L // block_size
    end_block = R // block_size
    
    # Якщо L і R в одному блоці
    if start_block == end_block:
        return sum(array[L:R + 1])
    
    # Додаємо залишок початкового блоку
    next_block_start = (start_block + 1) * block_size
    result += sum(array[L:next_block_start])
    
    # Додаємо повні блоки між початковим і кінцевим
    for block in range(start_block + 1, end_block):
        result += block_sums[block]
    
    # Додаємо початок кінцевого блоку
    end_block_start = end_block * block_size
    result += sum(array[end_block_start:R + 1])
    
    return result

def update_block_cache(index, value):
    block_id = index // block_size
    old_value = array[index]
    array[index] = value
    block_sums[block_id] += (value - old_value)

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_block_cache(query[1], query[2])
    elif query[0] == "Update":
        update_block_cache(query[1], query[2])
block_cache_time = time.time() - start_time


### 🔹 7. Комбінований метод: Блоки + LRU кеш
block_sums_combined = [sum(array[i:i + block_size]) for i in range(0, N, block_size)]

@lru_cache(maxsize=10000)
def range_sum_combined(L, R, counter):
    # Використовуємо ту ж логіку, що й у блочному кешуванні, але з LRU кешем
    result = 0
    start_block = L // block_size
    end_block = R // block_size
    
    if start_block == end_block:
        return sum(array[L:R + 1])
    
    next_block_start = (start_block + 1) * block_size
    result += sum(array[L:next_block_start])
    
    for block in range(start_block + 1, end_block):
        result += block_sums_combined[block]
    
    end_block_start = end_block * block_size
    result += sum(array[end_block_start:R + 1])
    
    return result

comb_invalidation_counter = 0

def update_combined(index, value):
    global comb_invalidation_counter
    block_id = index // block_size
    old_value = array[index]
    array[index] = value
    block_sums_combined[block_id] += (value - old_value)
    comb_invalidation_counter += 1

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_combined(query[1], query[2], comb_invalidation_counter)
    elif query[0] == "Update":
        update_combined(query[1], query[2])
combined_time = time.time() - start_time


### 🔹 8. Segment Tree (структура даних)
class SegmentTree:
    def __init__(self, array):
        self.n = len(array)
        self.tree = [0] * (4 * self.n)
        self.build(array, 0, 0, self.n - 1)

    def build(self, array, node, start, end):
        if start == end:
            self.tree[node] = array[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            self.build(array, left_child, start, mid)
            self.build(array, right_child, mid + 1, end)
            self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def update(self, idx, value, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            if idx <= mid:
                self.update(idx, value, left_child, start, mid)
            else:
                self.update(idx, value, right_child, mid + 1, end)
            self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def query(self, L, R, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1
        if R < start or L > end:
            return 0
        if L <= start and end <= R:
            return self.tree[node]
        mid = (start + end) // 2
        left_sum = self.query(L, R, 2 * node + 1, start, mid)
        right_sum = self.query(L, R, 2 * node + 2, mid + 1, end)
        return left_sum + right_sum

segment_tree = SegmentTree(array)

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        segment_tree.query(query[1], query[2])
    elif query[0] == "Update":
        segment_tree.update(query[1], query[2])
segment_tree_time = time.time() - start_time


### 🔹 9. Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def build(self, array):
        for i in range(1, self.n + 1):
            self.add(i, array[i - 1])

    def add(self, index, value):
        while index <= self.n:
            self.tree[index] += value
            index += index & -index

    def prefix_sum(self, index):
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result

    def query(self, L, R):
        return self.prefix_sum(R + 1) - self.prefix_sum(L)

    def update(self, index, value):
        diff = value - (self.query(index, index))
        self.add(index + 1, diff)

fenwick_tree = FenwickTree(N)
fenwick_tree.build(array)

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        fenwick_tree.query(query[1], query[2])
    elif query[0] == "Update":
        fenwick_tree.update(query[1], query[2])
fenwick_tree_time = time.time() - start_time


### 🔹 10. Sparse Table (тільки для запитів без оновлень)
# Sparse Table добре працює для запитів, але не підтримує оновлення
import math

class SparseTable:
    def __init__(self, array):
        self.n = len(array)
        self.log = int(math.log2(self.n)) + 1
        self.st = [[0] * self.n for _ in range(self.log + 1)]
        
        # Ініціалізуємо таблицю
        for i in range(self.n):
            self.st[0][i] = array[i]
        
        # Заповнюємо таблицю
        for i in range(1, self.log + 1):
            j = 0
            while j + (1 << i) <= self.n:
                self.st[i][j] = self.st[i-1][j] + self.st[i-1][j + (1 << (i-1))]
                j += 1

    def query(self, L, R):
        sum_result = 0
        length = R - L + 1
        for i in range(self.log, -1, -1):
            if (1 << i) <= length:
                sum_result += self.st[i][L]
                L += (1 << i)
                length -= (1 << i)
        return sum_result

    # Оновлення не ефективне в Sparse Table
    def update(self, index, value):
        array[index] = value
        # Перебудовуємо таблицю (не ефективно)
        self.__init__(array)

sparse_table = SparseTable(array)

start_time = time.time()
query_count = update_count = 0
for query in queries:
    if query[0] == "Range":
        sparse_table.query(query[1], query[2])
        query_count += 1
    elif query[0] == "Update":
        # Оновлення не ефективне в Sparse Table, тому ми просто модифікуємо array
        array[query[1]] = query[2]  # Це не впливає на Sparse Table
        update_count += 1
sparse_table_time = time.time() - start_time


### 📊 **Виведення результатів у гарному форматі**
methods = [
    "Без кешу (базовий)",
    "NumPy масив",
    "Префіксні суми",
    "Кеш з оптимізованим відстеженням",
    "Кеш з лічильником",
    "Блочне кешування",
    "Комбінований метод",
    "Segment Tree",
    "Fenwick Tree",
    "Sparse Table (тільки запити)"
]

times = [
    no_cache_time,
    numpy_time,
    prefix_sum_time,
    optimized_cache_time,
    counter_cache_time,
    block_cache_time,
    combined_time,
    segment_tree_time,
    fenwick_tree_time,
    sparse_table_time
]

# Розрахунок відносної швидкості відносно базового методу
relative_speeds = [no_cache_time / t if t > 0 else 0 for t in times]

# Оцінки складності
update_complexities = [
    "O(1)",
    "O(1)",
    "O(N)",
    "O(1) + інвалідація",
    "O(1)",
    "O(1)",
    "O(1)",
    "O(log N)",
    "O(log N)",
    "O(N) - неефективно"
]

query_complexities = [
    "O(R-L)",
    "O(R-L)",
    "O(1)",
    "O(1) з кешем, O(R-L) без",
    "O(1) з кешем, O(R-L) без",
    "O(sqrt(N))",
    "O(sqrt(N)) + кеш",
    "O(log N)",
    "O(log N)",
    "O(1)"
]

# Створення DataFrame
data = {
    "Метод": methods,
    "Час виконання (сек)": times,
    "Прискорення": [f"{speed:.2f}x" for speed in relative_speeds],
    "Оновлення": update_complexities,
    "Запит": query_complexities
}

df = pd.DataFrame(data)

# Сортуємо за часом виконання (від найшвидшого до найповільнішого)
df = df.sort_values("Час виконання (сек)")

# Функція для стилізації DataFrame з кольоровим градієнтом
def style_dataframe(df):
    # Копіюємо DataFrame для стилізації
    styling_df = df.copy()
    
    # Створюємо стилізований DataFrame
    styled_df = styling_df.style.background_gradient(
        cmap='RdYlGn_r',  # Червоний -> Жовтий -> Зелений (реверс)
        subset=['Час виконання (сек)'],
        vmin=min(times),
        vmax=max(times)
    )
    
    # Додаємо форматування
    styled_df = styled_df.format({
        'Час виконання (сек)': '{:.4f} сек',
    })
    
    # Додаємо заголовок
    styled_df = styled_df.set_caption(f"Порівняння методів (N={N:,}, Q={Q:,})")
    
    # Додаємо загальні стилі
    styled_df = styled_df.set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold')]},
        {'selector': 'th', 'props': [('font-size', '14px'), ('text-align', 'center'), 
                                    ('background-color', '#f0f0f0'), ('color', 'black')]},
        {'selector': 'td', 'props': [('font-size', '13px'), ('text-align', 'center')]}
    ])
    
    return styled_df

# Стилізуємо DataFrame
styled_df = style_dataframe(df)

# Додаємо інформацію про тести
print(f"\n{'='*80}")
print(f"{'ТЕСТУВАННЯ РІЗНИХ МЕТОДІВ ОБЧИСЛЕННЯ СУМ В МАСИВІ':^80}")
print(f"{'='*80}")
print(f"Розмір масиву: {N:,} елементів")
print(f"Кількість запитів: {Q:,} (приблизно {Q//2:,} запитів суми і {Q//2:,} оновлень)")
print(f"{'-'*80}")

# Виводимо найшвидший метод
fastest_method_idx = df['Час виконання (сек)'].idxmin()
fastest_method = df.loc[fastest_method_idx, 'Метод']
fastest_time = df.loc[fastest_method_idx, 'Час виконання (сек)']
print(f"Найшвидший метод: {fastest_method} ({fastest_time:.4f} сек)")

# Виводимо DataFrame
df