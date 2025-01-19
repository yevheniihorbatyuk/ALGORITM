### MapReduce

**MapReduce** --- це модель програмування для обробки та генерування великих наборів даних у розподілених системах. Вона дозволяє розділити обробку даних на паралельні підзадачі, що виконуються на кількох вузлах кластера, і автоматично об'єднує результати в кінцевий результат.

------------------------------------------------------------------------

### Ключові особливості MapReduce

1.  **Масштабованість:**
    - Обробка терабайтів і петабайтів даних за допомогою кластерів серверів.
2.  **Толерантність до збоїв:**
    - Алгоритм автоматично повторно виконує завдання, якщо один із вузлів виходить з ладу.
3.  **Паралелізація:**
    - Розбиває обробку на незалежні завдання, які виконуються паралельно.
4.  **Простота використання:**
    - Абстрагує складності розподілених обчислень, дозволяючи зосередитися на основній логіці.

------------------------------------------------------------------------

### Принцип роботи MapReduce

MapReduce складається з двох основних етапів:

#### 1. Map (карта):

- Кожен вхідний елемент перетворюється на пару **(ключ, значення)**.
- Завдання Map-функції --- розділити великий набір даних на проміжні результати.

#### 2. Reduce (зведення):

- Групує проміжні пари **(ключ, значення)** за ключами.
- Виконує операції агрегації, такі як підрахунок, сума, максимум/мінімум тощо, над значеннями, що мають однаковий ключ.

------------------------------------------------------------------------

### Алгоритм MapReduce (покроково)

#### Вхідні дані:

Розбиті на частини (наприклад, файли або блоки даних) і передаються на обробку.

1.  **Розбиття даних:**

    - Вхідні дані діляться на менші частини (шматки або блоки).
    - Наприклад, великий текстовий файл розділяється на кілька менших блоків.

2.  **Етап Map:**

    - Кожен блок передається на Map-функцію.
    - Map-функція обробляє дані та генерує проміжні пари **(ключ, значення)**.

3.  **Перестановка (Shuffle):**

    - Після виконання Map-функції проміжні пари сортуються та групуються за ключами.
    - Наприклад, всі пари з ключем `word1` групуються разом.

4.  **Етап Reduce:**

    - Зведення проміжних пар до кінцевого результату.
    - Наприклад, для ключа `word1` значення агрегуються (наприклад, підраховується їхня кількість).

5.  **Результат:**

    - Отримані результати записуються у вихідний файл або базу даних.

------------------------------------------------------------------------

### Приклад роботи MapReduce

#### Задача: Підрахунок слів у тексті

**Вхідні дані:**

    file1: "hello world"
    file2: "hello mapreduce"

#### Етап Map:

- Map-функція отримує рядки тексту і генерує пари **(слово, 1)**.

**Результат Map:**

    (hello, 1), (world, 1), (hello, 1), (mapreduce, 1)

#### Етап Shuffle:

- Групування пар за ключами.

**Результат Shuffle:**

    (hello, [1, 1]), (world, [1]), (mapreduce, [1])

#### Етап Reduce:

- Reduce-функція підсумовує значення для кожного ключа.

**Результат Reduce:**

    (hello, 2), (world, 1), (mapreduce, 1)

------------------------------------------------------------------------

### Реалізація на Python

``` python
from collections import defaultdict

# Map step
def map_function(data):
    result = []
    for line in data:
        for word in line.split():
            result.append((word, 1))  # Формуємо пари (слово, 1)
    return result

# Shuffle and sort
def shuffle_and_sort(mapped_data):
    grouped_data = defaultdict(list)
    for key, value in mapped_data:
        grouped_data[key].append(value)  # Групуємо за ключами
    return grouped_data

# Reduce step
def reduce_function(grouped_data):
    reduced_data = {}
    for key, values in grouped_data.items():
        reduced_data[key] = sum(values)  # Агрегація
    return reduced_data

# Input data
data = [
    "hello world",
    "hello mapreduce"
]

# Execute MapReduce
mapped_data = map_function(data)
grouped_data = shuffle_and_sort(mapped_data)
result = reduce_function(grouped_data)

print(result)  # Вихід: {'hello': 2, 'world': 1, 'mapreduce': 1}
```

------------------------------------------------------------------------

### Переваги MapReduce

1.  **Простота:**
    - Абстрагує складнощі паралельних обчислень.
2.  **Масштабованість:**
    - Легко масштабується для обробки великих обсягів даних на кластері.
3.  **Толерантність до збоїв:**
    - Автоматично відновлює завдання у разі помилок.

------------------------------------------------------------------------

### Недоліки MapReduce

1.  **Високі затримки:**
    - Кожен етап обробки (Map, Shuffle, Reduce) може займати багато часу.
2.  **Обмеження в гнучкості:**
    - Не підходить для всіх задач, наприклад, тих, що потребують ітерацій.
3.  **Надмірні операції запису/читання:**
    - Дані часто записуються на диск між етапами, що знижує продуктивність.

------------------------------------------------------------------------

### Застосування MapReduce

1.  **Пошукові системи:**

    - Аналіз веб-логів.
    - Індексування сторінок.

2.  **Big Data аналітика:**

    - Обробка транзакцій.
    - Аналіз клієнтської активності.

3.  **Машинне навчання:**

    - Попередня обробка великих наборів даних.

4.  **Геноміка:**

    - Аналіз послідовностей ДНК.

------------------------------------------------------------------------

### Висновок

MapReduce є потужним інструментом для роботи з великими даними, забезпечуючи масштабованість і толерантність до збоїв. Це ідеальний підхід для багатьох задач, які можуть бути розділені на незалежні підзадачі, і залишається основою сучасних технологій обробки даних, таких як Hadoop і Spark.

------------------------------------------------------------------------

### Функції `map` та `reduce` в контексті MapReduce

------------------------------------------------------------------------

#### `map` (карта)

- Функція `map` перетворює вхідні дані у пари **(ключ, значення)** для подальшої обробки.
- Це перший етап обробки в MapReduce, де дані розбиваються на менші частини і обробляються незалежно.
- **Задача:** Розділити вхідні дані так, щоб їх було легко агрегувати на наступному етапі.

##### Приклад: Підрахунок слів

Вхідний текст:

    "hello world hello mapreduce"

Результат `map`:

    [("hello", 1), ("world", 1), ("hello", 1), ("mapreduce", 1)]

------------------------------------------------------------------------

#### `reduce` (зведення)

- Функція `reduce` агрегує значення для однакових ключів, отриманих із попереднього етапу.
- Вона працює з групами даних, які мають однакові ключі, і обчислює підсумковий результат (сума, максимум, середнє тощо).

##### Приклад: Підрахунок слів

Вхідні дані (після `map`):

    [("hello", [1, 1]), ("world", [1]), ("mapreduce", [1])]

Результат `reduce`:

    [("hello", 2), ("world", 1), ("mapreduce", 1)]

------------------------------------------------------------------------

### Реалізація базової версії MapReduce

#### Приклад: Реалізація підрахунку слів вручну

``` python
from collections import defaultdict

# Функція map
def map_function(data):
    result = []
    for line in data:
        for word in line.split():
            result.append((word, 1))
    return result

# Функція shuffle (групування)
def shuffle(mapped_data):
    grouped_data = defaultdict(list)
    for key, value in mapped_data:
        grouped_data[key].append(value)
    return grouped_data

# Функція reduce
def reduce_function(grouped_data):
    reduced_data = {}
    for key, values in grouped_data.items():
        reduced_data[key] = sum(values)
    return reduced_data

# Вхідні дані
data = [
    "hello world",
    "hello mapreduce"
]

# Виконання MapReduce
mapped_data = map_function(data)
grouped_data = shuffle(mapped_data)
result = reduce_function(grouped_data)

print(result)  # {'hello': 2, 'world': 1, 'mapreduce': 1}
```

------------------------------------------------------------------------

#### Приклад: Використання PySpark

``` python
from pyspark import SparkContext

# Створення SparkContext
sc = SparkContext("local", "Word Count")

# Вхідні дані
data = ["hello world", "hello mapreduce"]

# Виконання MapReduce
rdd = sc.parallelize(data)
result = (rdd.flatMap(lambda line: line.split())  # Map: Розбиваємо рядки на слова
            .map(lambda word: (word, 1))         # Генеруємо пари (слово, 1)
            .reduceByKey(lambda a, b: a + b))    # Reduce: Підрахунок
print(result.collect())  # [('hello', 2), ('world', 1), ('mapreduce', 1)]
```

------------------------------------------------------------------------

### Паралельність у MapReduce

#### 1. Розбиття задачі на підзадачі

- Вхідні дані розбиваються на блоки (наприклад, текстовий файл на кілька частин).
- Кожна частина обробляється окремою функцією `map` на різних вузлах.

#### 2. Балансування навантаження

- Дані автоматично розподіляються між вузлами кластера.
- Якщо один вузол зайнятий або виходить з ладу, його завдання передається іншому вузлу.

#### 3. Локальність даних

- MapReduce виконує обробку даних якомога ближче до місця їхнього зберігання (наприклад, на тому ж сервері або жорсткому диску), щоб мінімізувати витрати на передачу даних.

------------------------------------------------------------------------

### Практичний приклад: Підрахунок унікальних транзакцій

#### Задача:

Обчислити кількість унікальних користувачів у великому наборі транзакцій.

Вхідні дані:

    user_id, amount
    1, 100
    2, 200
    1, 150
    3, 300
    2, 250

#### Реалізація MapReduce

``` python
# Функція map
def map_transactions(data):
    return [(line.split(",")[0], 1) for line in data]  # Витягуємо user_id

# Функція shuffle
def shuffle(mapped_data):
    grouped_data = defaultdict(list)
    for key, value in mapped_data:
        grouped_data[key].append(value)
    return grouped_data

# Функція reduce
def reduce_transactions(grouped_data):
    return {key: 1 for key in grouped_data.keys()}  # Унікальні user_id

# Вхідні дані
transactions = [
    "1,100",
    "2,200",
    "1,150",
    "3,300",
    "2,250"
]

# Виконання MapReduce
mapped_data = map_transactions(transactions)
grouped_data = shuffle(mapped_data)
unique_users = reduce_transactions(grouped_data)

print(f"Кількість унікальних користувачів: {len(unique_users)}")  # 3
```

------------------------------------------------------------------------

### **Висновок**

MapReduce забезпечує ефективне виконання паралельних обчислень, використовуючи два простих етапи: `map` і `reduce`. Цей підхід широко використовується для великих даних завдяки його масштабованості, надійності та простоті реалізації. Наведені приклади демонструють його потужність навіть у простих задачах, таких як підрахунок слів або аналіз транзакцій.

## Trade-offs

### Advantages 👍

1.  **Simplicity**
    - Abstract parallel computing
    - Focus on business logic
2.  **Scalability**
    - Easy cluster scaling
    - Automatic load distribution
3.  **Fault Tolerance**
    - Automatic recovery
    - Task replication

### Disadvantages 👎

1.  **Latency**
    - Processing stage delays
    - I/O overhead
2.  **Flexibility Limitations**
    - Not ideal for iterative tasks
    - Sequential dependency issues
3.  **I/O Intensive**
    - Disk operations between stages
    - Network transfer overhead

## Related Topics

- [Hadoop Ecosystem](Hadoop Ecosystem "wikilink")
- [Distributed Computing](Distributed Computing "wikilink")
- [Big Data Processing](Big Data Processing "wikilink")
- [Parallel Algorithms](Parallel Algorithms "wikilink")
- [Data Processing Frameworks](Data Processing Frameworks "wikilink")

## Additional Resources

- 📚 [MapReduce Patterns](MapReduce Patterns "wikilink")
- 🔧 [Implementation Guides](Implementation Guides "wikilink")
- 📊 [Performance Analysis](Performance Analysis "wikilink")
- 🎓 [Learning Resources](Learning Resources "wikilink")

\#big-data/frameworks \#distributed-computing \#parallel-processing \#data-processing \#programming/distributed-systems \#hadoop/mapreduce

------------------------------------------------------------------------

# Blended

## Детальне заняття на тему "MapReduce"

\#distributed-systems \#big-data \#mapreduce \#hadoop \#parallel-computing \#data-processing

## Ціль заняття

- Ознайомити студентів з концепцією та моделлю обчислень MapReduce
- Навчити реалізовувати базові задачі за допомогою MapReduce
- Пояснити, як використовувати MapReduce у розподілених системах (на прикладі Hadoop чи аналогів)
- Розширити знання студентів про паралельні обчислення та обробку великих даних

## План заняття

### 1. Теоретичний блок (20 хвилин)

#### 1.1 Що таке MapReduce?

- **Модель обчислень** для обробки великих обсягів даних у розподілених системах
- **Розділення роботи на дві основні стадії:**
  - **Map:** Перетворює вхідні дані у форму "ключ-значення"
  - **Reduce:** Агрегує результати з однаковими ключами

#### 1.2 Архітектура та робочий процес

- Дані розбиваються на **частини** (shards)
- **Маппери** обробляють кожну частину незалежно
- **Шафл (shuffle):** Розподіл даних за ключами для редьюсерів
- **Редьюсери:** Обчислюють результати для кожного ключа

#### 1.3 Приклади реальних задач

- Підрахунок частоти слів у текстових файлах
- Обробка веб-логів (підрахунок унікальних IP)
- Статистичний аналіз великих даних

#### 1.4 Ключові концепції

- Масштабованість: обробка великих даних на кластері машин
- Толерантність до відмов: автоматичне переназначення задач у випадку помилок
- Важливість ідентичності ключів у фазі "Shuffle and Sort"

#### 1.5 Обмеження

- Не завжди підходить для задач із складними залежностями між даними
- Високі накладні витрати на комунікацію між вузлами

#### Запитання для студентів

- Чому обробка ключ-значення є важливою для розподілених систем?
- Як "shuffle and sort" впливає на продуктивність MapReduce?

### 2. Практичний блок (70 хвилин)

#### 2.1 Базова реалізація MapReduce (30 хвилин)

**Завдання:** Написати код для підрахунку частоти слів у текстовому файлі за допомогою MapReduce.

``` python
from collections import defaultdict

# Map function
def map_function(lines):
    intermediate = []
    for line in lines:
        words = line.strip().split()
        for word in words:
            intermediate.append((word, 1))
    return intermediate

# Reduce function
def reduce_function(intermediate):
    result = defaultdict(int)
    for key, value in intermediate:
        result[key] += value
    return result

# Тестування
lines = [
    "big data is amazing",
    "big data needs tools like MapReduce",
    "data processing is key"
]

# Map phase
mapped = map_function(lines)

# Shuffle and Reduce phase
reduced = reduce_function(mapped)

# Виведення результатів
print("Результати підрахунку частоти слів:")
for word, count in reduced.items():
    print(f"{word}: {count}")
```

**Запитання для студентів:**
- Як об'єднати результат обчислення з кількох машин?
- Як оптимізувати обробку великих файлів?

#### 2.2 Реалізація MapReduce з використанням реальних даних (20 хвилин)

**Завдання:** Обробити веб-логи для підрахунку кількості запитів від кожного IP-адресу.

``` python
def map_logs(logs):
    intermediate = []
    for log in logs:
        ip = log.split()[0]
        intermediate.append((ip, 1))
    return intermediate

def reduce_logs(intermediate):
    result = defaultdict(int)
    for ip, count in intermediate:
        result[ip] += count
    return result

# Логи для тестування
logs = [
    "192.168.1.1 GET /index.html",
    "192.168.1.2 POST /login",
    "192.168.1.1 GET /about",
    "192.168.1.3 GET /index.html",
    "192.168.1.1 POST /logout"
]

mapped_logs = map_logs(logs)
reduced_logs = reduce_logs(mapped_logs)

print("Кількість запитів за IP:")
for ip, count in reduced_logs.items():
    print(f"{ip}: {count}")
```

**Запитання для студентів:**
- Як можна змінити формат логів, щоб отримати більше інформації?
- Як забезпечити паралельне виконання цього алгоритму?

#### 2.3 Розподілена обробка з Hadoop (20 хвилин)

**Завдання:** Розібрати робочий процес MapReduce у Hadoop через інтерфейс.

``` python
# Імітація Hadoop через Python
from multiprocessing import Pool

def parallel_map(mapper, chunks):
    with Pool(len(chunks)) as pool:
        return pool.map(mapper, chunks)

def parallel_reduce(reducer, mapped_chunks):
    combined = []
    for chunk in mapped_chunks:
        combined.extend(chunk)
    return reducer(combined)

lines = [
    "big data is amazing",
    "big data needs tools like MapReduce",
    "data processing is key"
]

# Поділ вхідних даних на шматки
chunks = [lines[:2], lines[2:]]

# Виконання MapReduce у паралельному режимі
mapped_chunks = parallel_map(map_function, chunks)
result = parallel_reduce(reduce_function, mapped_chunks)

print("Результат паралельного MapReduce:")
for word, count in result.items():
    print(f"{word}: {count}")
```

**Запитання для студентів:**
- Як Hadoop обробляє невдалі задачі (failover)?
- Які обмеження Hadoop порівняно з сучасними системами (наприклад, Apache Spark)?

### 3. Рефлексія та обговорення (10 хвилин)

1.  Які типи задач добре підходять для моделі MapReduce?
2.  Як можна покращити ефективність "shuffle and sort"?
3.  Як ця модель використовується у сучасних платформах (наприклад, Spark, Flink)?

## Список релевантних концепцій/технологій/тем для доповнення

\#related-topics
- **Hadoop:** Архітектура і робота з HDFS (Hadoop Distributed File System)
- **Порівняння:** MapReduce vs Spark (переваги та недоліки кожної системи)
- **Паралельні обчислення:** Принципи та виклики
- **Apache Beam:** Універсальна модель обробки даних, сумісна з MapReduce
- **Batch vs Stream Processing:** Як моделі обчислень відрізняються у різних сценаріях

## Домашнє завдання

### 1. Теоретичне питання

- Доведіть, чому MapReduce масштабовано навіть для мільярдів записів
- Як "combiner" може зменшити навантаження на редьюсери?

### 2. Практичне завдання

- Реалізувати MapReduce для підрахунку популярності продуктів у даних продажів (формат: `product_id, quantity`)
- Використати Hadoop (або PySpark) для обробки великих текстових файлів

\#homework \#practice \#theory

# Questions

## List

**Теоретичний блок:**

- Чому обробка ключ-значення є важливою для розподілених систем?
- Як "shuffle and sort" впливає на продуктивність MapReduce?

**Практичний блок (Запитання для студентів до першого завдання):**

- Як об'єднати результат обчислення з кількох машин?
- Як оптимізувати обробку великих файлів?

**Практичний блок (Запитання для студентів до другого завдання):**

- Як можна змінити формат логів, щоб отримати більше інформації?
- Як забезпечити паралельне виконання цього алгоритму?

**Практичний блок (Запитання для студентів до третього завдання):**

- Як Hadoop обробляє невдалі задачі (failover)?
- Які обмеження Hadoop порівняно з сучасними системами (наприклад, Apache Spark)?

**Рефлексія та обговорення (після практичного блоку):**

- Які типи задач добре підходять для моделі MapReduce?
- Як можна покращити ефективність "shuffle and sort"?
- Як ця модель використовується у сучасних платформах (наприклад, Spark, Flink)?

**Домашнє завдання (Теоретичне питання):**

- Доведіть, чому MapReduce масштабовано навіть для мільярдів записів.
- Як "combiner" може зменшити навантаження на редьюсери?

**Домашнє завдання (Практичне завдання):**

- Реалізувати MapReduce для підрахунку популярності продуктів у даних продажів (формат: `product_id, quantity`).
- Використати Hadoop (або PySpark) для обробки великих текстових файлів.

# MapReduce: Way Out

\#distributed-computing \#big-data \#mapreduce \#parallel-processing

## Теоретичний блок

### Чому обробка ключ-значення є важливою для розподілених систем?

\#key-value \#distributed-systems

**Роз'яснення:** Обробка у форматі "ключ-значення" дозволяє:

1.  **Розподілити дані:** Ключі використовуються для шардінгу даних між вузлами, забезпечуючи паралельну обробку
2.  **Агрегацію даних:** Згруповані за ключами значення обробляються незалежно, що мінімізує залежності між вузлами
3.  **Простоту реалізації:** Формат забезпечує модульність та універсальність обробки даних у багатьох сценаріях

**Реальне застосування:**
- Підрахунок популярності продуктів за категоріями
- Агрегація логів для аналізу мережевого трафіку

**Технології:**
- **Hadoop:** Використовує "ключ-значення" для поділу задач між мапперами та редьюсерами
- **Cassandra:** Побудована на моделі "ключ-значення" для розподіленого зберігання даних

### Як "shuffle and sort" впливає на продуктивність MapReduce?

\#performance \#optimization

**Роз'яснення:** Процес "shuffle and sort":

1.  **Розподіляє дані:** Переносить дані між мапперами і редьюсерами, забезпечуючи агрегацію даних з однаковими ключами
2.  **Оптимізує обчислення:** Сортування дозволяє редьюсерам ефективно працювати з послідовними ключами
3.  **Впливає на продуктивність:** Це найдовша стадія через необхідність пересилання великих обсягів даних

**Технології:**
- **Apache Tez:** Альтернатива MapReduce для оптимізації "shuffle and sort"
- **Spark:** Використовує пам'ять для зменшення витрат на сортування

## Практична реалізація

### Базовий приклад MapReduce для обробки продажів

``` python
from typing import List, Tuple, Dict
from collections import defaultdict

def map_sales(data: List[str]) -> List[Tuple[str, int]]:
    """
    Map функція для обробки даних продажів.
    
    Args:
        data: Список рядків з даними продажів у форматі "product,quantity"
        
    Returns:
        List[Tuple[str, int]]: Список пар (продукт, кількість)
    """
    intermediate = []
    for entry in data:
        product, quantity = entry.split(",")
        intermediate.append((product, int(quantity)))
    return intermediate

def reduce_sales(intermediate: List[Tuple[str, int]]) -> Dict[str, int]:
    """
    Reduce функція для агрегації продажів.
    
    Args:
        intermediate: Список пар (продукт, кількість)
    
    Returns:
        Dict[str, int]: Словник з загальною кількістю продажів по продуктах
    """
    result = defaultdict(int)
    for product, quantity in intermediate:
        result[product] += quantity
    return dict(result)

# Приклад використання
sales_data = [
    "product1,2",
    "product2,3",
    "product1,5",
    "product3,7",
    "product2,1"
]

mapped_sales = map_sales(sales_data)
reduced_sales = reduce_sales(mapped_sales)

print("Продажі продуктів:")
for product, total in reduced_sales.items():
    print(f"{product}: {total}")
```

### Реалізація у PySpark для великих даних

``` python
from pyspark.sql import SparkSession
from typing import List, Tuple

def process_large_file(input_path: str, output_path: str) -> None:
    """
    Обробляє великий текстовий файл використовуючи PySpark.
    
    Args:
        input_path: Шлях до вхідного файлу
        output_path: Шлях для збереження результатів
    """
    spark = SparkSession.builder \
        .appName("WordCount") \
        .getOrCreate()

    # Читаємо файл
    text_file = spark.sparkContext.textFile(input_path)

    # MapReduce операції
    word_counts = text_file \
        .flatMap(lambda line: line.split()) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b)

    # Зберігаємо результат
    word_counts.saveAsTextFile(output_path)
    
    spark.stop()
```

## Розподілена обробка та відмовостійкість

### Механізми обробки невдалих задач

\#fault-tolerance \#reliability

1.  **Автоматичне переназначення:**
    - Задачі переназначаються на інші вузли при відмові
    - HDFS забезпечує надійне зберігання проміжних результатів
2.  **Моніторинг стану:**
    - JobTracker (Hadoop 1) або ResourceManager (Hadoop 2)
    - Постійна перевірка здоров'я вузлів кластера

### Оптимізація продуктивності

\#performance-optimization

1.  **Комбінери (Combiners):**
    - Локальна агрегація на вузлах мапперів
    - Зменшення мережевого трафіку
2.  **Партиціонування даних:**
    - Оптимальний розподіл ключів між редьюсерами
    - Уникнення дисбалансу навантаження

## Порівняння з сучасними системами

### MapReduce vs Spark

\#comparison \#modern-systems

**MapReduce переваги:**
- Простота моделі
- Надійність для batch-обробки
- Висока відмовостійкість

**Spark переваги:**
- Обробка в пам'яті
- Підтримка стрімінгу
- Багатший API

## Підсумок

MapReduce залишається фундаментальною моделлю для розподілених обчислень, особливо для:
1. Обробки великих обсягів даних
2. ETL-процесів
3. Аналітичних задач

\#mapreduce \#hadoop \#spark \#distributed-computing \#big-data \#data-engineering
