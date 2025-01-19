---
tags:
- algorithms/probabilistic
- data-structures/streaming
- big-data/cardinality
- algorithms/estimation
- computer-science/algorithms
- data-processing/streaming
---

# HyperLogLog Algorithm

## Overview

**HyperLogLog** is a probabilistic algorithm for estimating the number of **unique elements** in a large dataset or stream. It's extremely memory-efficient, allowing for processing of large data volumes with accurate estimation using minimal memory.

## Core Problem

Estimation of set cardinality $|A|$ (number of unique elements in dataset $A$) without storing all elements.

## Key Advantages

### 1. Memory Efficiency 💾

- Uses approximately $1.5 \log_2(\log_2(n))$ bits
- For 1 billion elements, only needs 1.5 KB of memory
- 🔗 [Memory Efficient Algorithms](Memory Efficient Algorithms "wikilink")

### 2. Processing Speed ⚡

- Operates in streaming mode
- $O(1)$ complexity for adding elements
- 🔗 [Stream Processing](Stream Processing "wikilink")

### 3. Probabilistic Nature 🎲

- High accuracy with controllable deviation
- 🔗 [Probabilistic Algorithms](Probabilistic Algorithms "wikilink")

## Working Principle

### 1. Core Concept

The algorithm uses hashing to transform elements into numerical values. Instead of storing unique values, HyperLogLog analyzes hash distributions for cardinality estimation.

### 2. Hash Distribution

1.  Uses hash function $h(x)$ mapping elements to large range ($h: x \to \{0, 1\}^L$)
2.  Analyzes leading zeros in hash:
    - Example: $h(x_1) = 001011$ (two leading zeros)
    - Example: $h(x_2) = 000110$ (three leading zeros)
3.  Number of leading zeros indicates set size

### 3. Data Partitioning

- Divides hash space into $m$ levels (registers)
- Each register analyzes part of the stream
- 🔗 [Data Partitioning](Data Partitioning "wikilink")

## Algorithm Steps

### 1. Element Hashing 🔄

- Convert each element $x$ to $h(x)$

### 2. Register Division

- Split into $m$ registers using lower bits
- Use upper bits for zero counting

### 3. Prefix Counting

- Store maximum leading zeros per register

### 4. Size Estimation

- Use harmonic mean with correction factors

## Mathematical Foundation

### Cardinality Estimation

The main formula:
$E = \alpha_m \cdot m^2 \cdot \left(\sum_{j=1}^m 2^{-M[j]}\right)^{-1}$

Where:
- $m$: register count
- $M[j]$: maximum zeros in register $j$
- $\alpha_m$: correction factor based on $m$

### Corrections

1.  **Small Sets**: Linear correction when $E < 2.5m$
2.  **Large Sets**: Logarithmic correction when $E > 2^{32}$

## Implementation

``` python
import hashlib
import math

class HyperLogLog:
    def __init__(self, b):
        self.b = b  # Number of bits for registers
        self.m = 1 << b  # Number of registers
        self.registers = [0] * self.m
        self.alpha_m = self._get_alpha_m()

    def _get_alpha_m(self):
        if self.m == 16:
            return 0.673
        elif self.m == 32:
            return 0.697
        elif self.m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / self.m)

    def _hash(self, value):
        h = hashlib.md5(value.encode('utf8')).hexdigest()
        return int(h, 16)

    def add(self, value):
        x = self._hash(value)
        j = x & (self.m - 1)  # Register determination
        w = x >> self.b  # Upper bits
        self.registers[j] = max(self.registers[j], self._rho(w))

    def _rho(self, w):
        return len(bin(w)) - len(bin(w).rstrip('0'))

    def estimate(self):
        Z = sum(2.0 ** -r for r in self.registers)
        E = self.alpha_m * self.m ** 2 / Z

        # Small range correction
        if E <= 2.5 * self.m:
            V = self.registers.count(0)
            if V > 0:
                E = self.m * math.log(self.m / V)

        # Large range correction
        if E > 1 / 30.0 * 2 ** 32:
            E = -(2 ** 32) * math.log(1 - E / 2 ** 32)

        return E
```

## Practical Applications

### 1. Big Data Analytics 📊

- Unique user counting
- Real-time interaction analysis
- 🔗 [Big Data Analytics](Big Data Analytics "wikilink")

### 2. Databases 💾

- Index optimization
- Unique key estimation
- 🔗 [Database Optimization](Database Optimization "wikilink")

### 3. Search Systems 🔍

- Unique result counting
- Query optimization
- 🔗 [Search Optimization](Search Optimization "wikilink")

### 4. Network Traffic 🌐

- Connection analysis
- Packet monitoring
- 🔗 [Network Analysis](Network Analysis "wikilink")

## Related Topics

- [Probabilistic Data Structures](Probabilistic Data Structures "wikilink")
- [Stream Processing Algorithms](Stream Processing Algorithms "wikilink")
- [Cardinality Estimation](Cardinality Estimation "wikilink")
- [Hash Functions](Hash Functions "wikilink")
- [Big Data Processing](Big Data Processing "wikilink")

## Additional Resources

- 📚 [Probabilistic Algorithms Theory](Probabilistic Algorithms Theory "wikilink")
- 🔧 [Stream Processing Tools](Stream Processing Tools "wikilink")
- 📊 [Data Estimation Methods](Data Estimation Methods "wikilink")
- 🎓 [Algorithm Analysis](Algorithm Analysis "wikilink")

\#algorithms/probabilistic \#data-structures/streaming \#big-data/cardinality \#algorithms/estimation \#computer-science/algorithms \#data-processing/streamingata-processing/streaming

------------------------------------------------------------------------

# Blended

## Детальне заняття на тему "HyperLogLog"

\#algorithms \#data-structures \#computer-science \#hyperloglog \#big-data \#probability \#cardinality-estimation

## Ціль заняття

- Ознайомити студентів із алгоритмом HyperLogLog (HLL) для підрахунку унікальних елементів у потоці даних
- Розглянути теоретичну основу, ключові концепції та практичну імплементацію
- Навчити застосовувати HLL для задач реального світу та ефективно налаштовувати параметри алгоритму

## План заняття

### 1. Теоретичний блок (15 хвилин)

#### 1.1 Що таке HyperLogLog?

- HyperLogLog --- це алгоритм, призначений для приблизного підрахунку кількості унікальних елементів у великому потоці даних
- Використовує фіксований обсяг пам'яті, незалежно від розміру вхідних даних

#### 1.2 Основна ідея алгоритму

- Використання хеш-функцій для рівномірного розподілу елементів
- Запис значень у "регістрах" за допомогою визначення найбільшого порядку нулів у хешованих значеннях
- Оцінка кількості унікальних елементів через гармонійне середнє

#### 1.3 Ключові компоненти

- **Хеш-функція:** Перетворює кожен елемент у числове значення
- **Регістри:** Масив, який зберігає позицію найбільшого біта (префікс нулів)
- **Оцінка кількості унікальних елементів:** Використовує формулу:
  $$E = \alpha_m \cdot m^2 \cdot \left(\sum_{i=1}^m 2^{-R[i]}\right)^{-1}$$
  де $m$ --- кількість регістрів, $R[i]$ --- максимальна кількість нулів у $i$-му регістрі, а $\alpha_m$ --- коригувальний коефіцієнт

#### 1.4 Обмеження

- Алгоритм не повертає точну кількість унікальних значень, але має низьку похибку ($\approx 1.04 / \sqrt{m}$)
- Неможливість видалення елементів

#### 1.5 Реальне застосування

- Аналіз трафіку (підрахунок унікальних IP-адрес)
- Системи рекомендацій (облік унікальних користувачів)
- Аналітика великих даних

#### Запитання для студентів

- Чому алгоритм використовує хешування, а не зберігає самі елементи?
- Як HLL справляється з великими потоками даних при фіксованій пам'яті?

### 2. Практичний блок (70 хвилин)

#### 2.1 Реалізація базового HyperLogLog (20 хвилин)

**Завдання:** Реалізувати базову версію HLL для підрахунку унікальних елементів у Python.

``` python
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

# Тестування
hll = HyperLogLog(10)  # Використовуємо 2^10 регістрів
elements = [f"user_{i}" for i in range(1, 10000)]

for elem in elements:
    hll.add(elem)

print("Оцінка кількості унікальних елементів:", round(hll.estimate()))
```

**Запитання для студентів:**
- Як зміниться точність оцінки при збільшенні $b$?
- Чи можливо використовувати інші хеш-функції?

#### 2.2 Налаштування параметрів HLL (20 хвилин)

**Завдання:** Дослідити вплив параметра $b$ на точність і обсяг пам'яті.

``` python
def analyze_hll_accuracy():
    for b in range(4, 16):
        hll = HyperLogLog(b)
        elements = [f"user_{i}" for i in range(1, 10000)]
        for elem in elements:
            hll.add(elem)
        print(f"b={b}, Регістрів={hll.m}, Оцінка={round(hll.estimate())}")

analyze_hll_accuracy()
```

**Обговорення:**
- Чому більше регістрів покращує точність?
- Який $b$ підходить для 1 мільйона унікальних елементів?

#### 2.3 Реальне застосування: Підрахунок унікальних IP-адрес (30 хвилин)

**Завдання:** Використати HLL для підрахунку унікальних IP-адрес зі згенерованого потоку.

``` python
import random

def generate_ip_addresses(count):
    return [f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(count)]

# Тестування
ip_addresses = generate_ip_addresses(100000)
hll = HyperLogLog(12)  # Для великого потоку

for ip in ip_addresses:
    hll.add(ip)

print("Кількість унікальних IP-адрес:", round(hll.estimate()))
```

**Додаткове завдання:**
- Порівняти оцінку HLL із точною кількістю за допомогою `set`

**Обговорення:**
- У яких ситуаціях можна використовувати HLL замість точних методів?
- Чи можна застосувати HLL для об'єднання кількох потоків?

### 3. Рефлексія та обговорення (10 хвилин)

1.  Які переваги та недоліки HLL порівняно із звичайним підрахунком?
2.  Як впливає кількість регістрів на продуктивність та пам'ять?
3.  Чи можна адаптувати HLL для задач із видаленням елементів?

## Список релевантних концепцій/технологій/тем для доповнення

\#related-topics
- **Хеш-функції:** Розгляд різних варіантів (MurmurHash, SHA-256)
- **Linear Counting:** Альтернативний метод для підрахунку унікальних елементів
- **Комбінація HLL із іншими алгоритмами:** Наприклад, використання разом із Count-Min Sketch
- **Великі дані:** Застосування HLL у хмарних платформах (наприклад, Redis HyperLogLog)
- **Об'єднання HLL:** Використання властивості об'єднання для агрегації даних

## Домашнє завдання

### 1. Теоретичне питання

- Доведіть, чому HLL має похибку $\approx 1.04 / \sqrt{m}$
- Як можна покращити точність HLL без збільшення регістрів?

### 2. Практичне завдання

- Реалізувати метод об'єднання кількох HLL-структур
- Використати HLL для підрахунку унікальних елементів у реальному наборі даних (напр., CSV-файл)

\#homework \#practice \#theory

# Blended

## Детальне заняття на тему "HyperLogLog"

\#algorithms \#data-structures \#computer-science \#hyperloglog \#big-data \#probability \#cardinality-estimation

## Ціль заняття

- Ознайомити студентів із алгоритмом HyperLogLog (HLL) для підрахунку унікальних елементів у потоці даних
- Розглянути теоретичну основу, ключові концепції та практичну імплементацію
- Навчити застосовувати HLL для задач реального світу та ефективно налаштовувати параметри алгоритму

## План заняття

### 1. Теоретичний блок (15 хвилин)

#### 1.1 Що таке HyperLogLog?

- HyperLogLog --- це алгоритм, призначений для приблизного підрахунку кількості унікальних елементів у великому потоці даних
- Використовує фіксований обсяг пам'яті, незалежно від розміру вхідних даних

#### 1.2 Основна ідея алгоритму

- Використання хеш-функцій для рівномірного розподілу елементів
- Запис значень у "регістрах" за допомогою визначення найбільшого порядку нулів у хешованих значеннях
- Оцінка кількості унікальних елементів через гармонійне середнє

#### 1.3 Ключові компоненти

- **Хеш-функція:** Перетворює кожен елемент у числове значення
- **Регістри:** Масив, який зберігає позицію найбільшого біта (префікс нулів)
- **Оцінка кількості унікальних елементів:** Використовує формулу:
  $$E = \alpha_m \cdot m^2 \cdot \left(\sum_{i=1}^m 2^{-R[i]}\right)^{-1}$$
  де $m$ --- кількість регістрів, $R[i]$ --- максимальна кількість нулів у $i$-му регістрі, а $\alpha_m$ --- коригувальний коефіцієнт

#### 1.4 Обмеження

- Алгоритм не повертає точну кількість унікальних значень, але має низьку похибку ($\approx 1.04 / \sqrt{m}$)
- Неможливість видалення елементів

#### 1.5 Реальне застосування

- Аналіз трафіку (підрахунок унікальних IP-адрес)
- Системи рекомендацій (облік унікальних користувачів)
- Аналітика великих даних

#### Запитання для студентів

- Чому алгоритм використовує хешування, а не зберігає самі елементи?
- Як HLL справляється з великими потоками даних при фіксованій пам'яті?

### 2. Практичний блок (70 хвилин)

#### 2.1 Реалізація базового HyperLogLog (20 хвилин)

**Завдання:** Реалізувати базову версію HLL для підрахунку унікальних елементів у Python.

``` python
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

# Тестування
hll = HyperLogLog(10)  # Використовуємо 2^10 регістрів
elements = [f"user_{i}" for i in range(1, 10000)]

for elem in elements:
    hll.add(elem)

print("Оцінка кількості унікальних елементів:", round(hll.estimate()))
```

**Запитання для студентів:**
- Як зміниться точність оцінки при збільшенні $b$?
- Чи можливо використовувати інші хеш-функції?

#### 2.2 Налаштування параметрів HLL (20 хвилин)

**Завдання:** Дослідити вплив параметра $b$ на точність і обсяг пам'яті.

``` python
def analyze_hll_accuracy():
    for b in range(4, 16):
        hll = HyperLogLog(b)
        elements = [f"user_{i}" for i in range(1, 10000)]
        for elem in elements:
            hll.add(elem)
        print(f"b={b}, Регістрів={hll.m}, Оцінка={round(hll.estimate())}")

analyze_hll_accuracy()
```

**Обговорення:**
- Чому більше регістрів покращує точність?
- Який $b$ підходить для 1 мільйона унікальних елементів?

#### 2.3 Реальне застосування: Підрахунок унікальних IP-адрес (30 хвилин)

**Завдання:** Використати HLL для підрахунку унікальних IP-адрес зі згенерованого потоку.

``` python
import random

def generate_ip_addresses(count):
    return [f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(count)]

# Тестування
ip_addresses = generate_ip_addresses(100000)
hll = HyperLogLog(12)  # Для великого потоку

for ip in ip_addresses:
    hll.add(ip)

print("Кількість унікальних IP-адрес:", round(hll.estimate()))
```

**Додаткове завдання:**
- Порівняти оцінку HLL із точною кількістю за допомогою `set`

**Обговорення:**
- У яких ситуаціях можна використовувати HLL замість точних методів?
- Чи можна застосувати HLL для об'єднання кількох потоків?

### 3. Рефлексія та обговорення (10 хвилин)

1.  Які переваги та недоліки HLL порівняно із звичайним підрахунком?
2.  Як впливає кількість регістрів на продуктивність та пам'ять?
3.  Чи можна адаптувати HLL для задач із видаленням елементів?

## Список релевантних концепцій/технологій/тем для доповнення

\#related-topics
- **Хеш-функції:** Розгляд різних варіантів (MurmurHash, SHA-256)
- **Linear Counting:** Альтернативний метод для підрахунку унікальних елементів
- **Комбінація HLL із іншими алгоритмами:** Наприклад, використання разом із Count-Min Sketch
- **Великі дані:** Застосування HLL у хмарних платформах (наприклад, Redis HyperLogLog)
- **Об'єднання HLL:** Використання властивості об'єднання для агрегації даних

## Домашнє завдання

### 1. Теоретичне питання

- Доведіть, чому HLL має похибку $\approx 1.04 / \sqrt{m}$
- Як можна покращити точність HLL без збільшення регістрів?

### 2. Практичне завдання

- Реалізувати метод об'єднання кількох HLL-структур
- Використати HLL для підрахунку унікальних елементів у реальному наборі даних (напр., CSV-файл)

\#homework \#practice \#theory

# Questions

## List

**Теоретичний блок:**

- Чому алгоритм використовує хешування, а не зберігає самі елементи?
- Як HLL справляється з великими потоками даних при фіксованій пам'яті?

**Практичний блок (Запитання для студентів до першого завдання):**

- Як зміниться точність оцінки при збільшенні b?
- Чи можливо використовувати інші хеш-функції?

**Практичний блок (Обговорення до другого завдання):**

- Чому більше регістрів покращує точність?
- Який b підходить для 1 мільйона унікальних елементів?

**Практичний блок (Обговорення до третього завдання):**

- У яких ситуаціях можна використовувати HLL замість точних методів?
- Чи можна застосувати HLL для об'єднання кількох потоків?

**Рефлексія та обговорення (після практичного блоку):**

- Які переваги та недоліки HLL порівняно із звичайним підрахунком?
- Як впливає кількість регістрів на продуктивність та пам'ять?
- Чи можна адаптувати HLL для задач із видаленням елементів?

**Домашнє завдання (Теоретичне питання):**

- Доведіть, чому HLL має похибку ≈1.04
- Як можна покращити точність HLL без збільшення регістрів?

**Домашнє завдання (Практичне завдання):**

- Реалізувати метод об'єднання кількох HLL-структур

- Використати HLL для підрахунку унікальних елементів у реальному наборі даних (напр., CSV-файл)

- # HyperLogLog: Way Out

\#algorithms \#probabilistic \#cardinality-estimation \#big-data

## Теоретичний блок

### Чому алгоритм використовує хешування, а не зберігає самі елементи?

\#hashing \#memory-optimization

**Роз'яснення:** HyperLogLog використовує хешування для компактного представлення даних, що дозволяє:

1.  **Знизити обсяг пам'яті:** Замість зберігання всіх елементів, зберігаються лише результати хешування
2.  **Забезпечити рівномірний розподіл:** Хешування розподіляє елементи по регістрах, дозволяючи точно оцінювати кількість унікальних значень

**Реальне застосування:**
- Підрахунок унікальних користувачів на веб-сайтах
- Моніторинг мережевого трафіку для унікальних IP-адрес

**Технології:**
- **Redis HyperLogLog:** Використовує вбудовані хеш-функції для роботи з великими множинами
- **BigQuery:** Застосовує HLL для оптимізації запитів до великих таблиць

### Як HLL справляється з великими потоками даних при фіксованій пам'яті?

\#streaming \#memory-efficiency

**Роз'яснення:** HyperLogLog працює з фіксованою кількістю регістрів ($m = 2^b$), що дозволяє обробляти великі потоки даних без збільшення пам'яті. Хеш-функції переводять елементи в 32-бітні числа, які використовуються для оновлення регістрів.

**Реальне застосування:**
- Системи потокової обробки даних (Apache Flink, Kafka Streams) для оцінки унікальних подій у реальному часі

**Технології:**
- **Apache Druid:** Використовує HLL для обчислення кардинальності у великих потоках аналітичних даних

## Практичний блок

### Як зміниться точність оцінки при збільшенні $b$?

\#accuracy \#parameter-tuning

**Роз'яснення:** Збільшення $b$ (кількості бітів для індексації) збільшує кількість регістрів ($m = 2^b$), що зменшує похибку оцінки. Похибка алгоритму обернено пропорційна $\sqrt{m}$:

$$\text{Похибка} \approx \frac{1.04}{\sqrt{m}}$$

**Реальне застосування:**
- При роботі з дуже великими множинами даних, наприклад, підрахунок унікальних відвідувачів у масштабних рекламних кампаніях

### Чи можливо використовувати інші хеш-функції?

\#hash-functions \#implementation

**Роз'яснення:** Так, можливо. Вибір хеш-функції впливає на рівномірність розподілу даних. Для HLL важливо, щоб хеш-функція:

1.  **Мала хороші стохастичні властивості**
2.  **Не створювала колізій**

**Технології:**
- **Python libraries:** `mmh3` (MurmurHash), `hashlib` (SHA-256)

### Доведення похибки HLL

\#mathematical-proof \#error-analysis

Похибка HLL визначається формулою:
$$\text{Похибка} = \Theta\left(\frac{1}{\sqrt{m}}\right)$$

**Доведення:**
1. Формула оцінки кардинальності:
$$E = \alpha_m \cdot m^2 \cdot \left(\sum_{i=1}^m 2^{-R[i]}\right)^{-1}$$
де $\alpha_m$ --- коригувальний коефіцієнт

2.  Експериментально встановлено, що оптимальний коефіцієнт дорівнює 1.04

### Практична реалізація об'єднання HLL-структур

``` python
class HyperLogLog:
    def merge(self, other_hll):
        """
        Об'єднує два HyperLogLog лічильники.
        
        Args:
            other_hll: Інший HyperLogLog для об'єднання
            
        Raises:
            ValueError: Якщо розміри регістрів не збігаються
        """
        if self.m != other_hll.m:
            raise ValueError("Розмір регістрів має збігатися")
            
        self.registers = [
            max(self.registers[i], other_hll.registers[i]) 
            for i in range(self.m)
        ]
```

### Обробка реальних даних з CSV

``` python
import csv

def analyze_unique_values(file_path: str, column_name: str, precision: int = 12) -> int:
    """
    Аналізує унікальні значення в CSV файлі використовуючи HyperLogLog.
    
    Args:
        file_path: Шлях до CSV файлу
        column_name: Назва колонки для аналізу
        precision: Точність HLL (кількість бітів)
    
    Returns:
        int: Оцінка кількості унікальних значень
    """
    hll = HyperLogLog(precision)
    
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            hll.add(row[column_name])
            
    return round(hll.estimate())
```

## Підсумок

HyperLogLog є потужним алгоритмом для оцінки кардинальності множин з такими ключовими перевагами:

1.  Фіксоване використання пам'яті
2.  Висока точність оцінки
3.  Можливість об'єднання структур
4.  Ефективність у розподілених системах

\#hyperloglog \#algorithms \#big-data \#streaming \#probabilistic-algorithms \#data-structures \#distributed-computing
