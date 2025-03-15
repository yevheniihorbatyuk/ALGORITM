import logging
import time
import os
import copy
import numpy as np
from typing import Dict, List, Callable, Any, Union, Tuple, Optional, TypeVar
from core.optimizer import Optimizer, OptimizerFactory
import pandas as pd
import dask.dataframe as dd


T = TypeVar('T')  # Тип стану (наприклад, маршрут або розподіл посилок)

class DaskExecutor:
    """Клас для обробки великих обсягів даних з використанням Dask."""
    
    def __init__(
        self, 
        use_local: bool = False, 
        num_workers: int = None,
        memory_limit: str = "4GB",
        scheduler_address: str = None,
        initialize_client: bool = True,
        verbose: bool = False
    ):
        """
        Ініціалізує Dask-виконавця.
        
        Args:
            use_local: Якщо True, виконує операції локально (без Dask)
            num_workers: Кількість воркерів Dask (якщо None - автоматично)
            memory_limit: Ліміт пам'яті на одного воркера
            scheduler_address: Адреса планувальника Dask (якщо використовується зовнішній кластер)
            initialize_client: Якщо True, ініціалізує клієнт під час створення об'єкта
            verbose: Режим детального логування
        """
        self.use_local = use_local
        self.num_workers = num_workers
        self.memory_limit = memory_limit
        self.scheduler_address = scheduler_address
        self.verbose = verbose
        self.client = None
        
        if not use_local and initialize_client:
            self._initialize_client()
    
    def _initialize_client(self):
        """Ініціалізує Dask Client."""
        try:
            from dask.distributed import Client, LocalCluster
            
            if self.scheduler_address:
                # Підключаємося до існуючого кластера
                if self.verbose:
                    print(f"Підключення до кластера Dask за адресою: {self.scheduler_address}")
                self.client = Client(self.scheduler_address)
            else:
                # Створюємо локальний кластер
                if self.verbose:
                    print(f"Створення локального Dask кластера з {self.num_workers or 'автовизначеною'} кількістю воркерів")
                cluster = LocalCluster(
                    n_workers=self.num_workers,
                    threads_per_worker=1,
                    memory_limit=self.memory_limit
                )
                self.client = Client(cluster)
                
            if self.verbose:
                print(f"Dask Dashboard URL: {self.client.dashboard_link}")
        except ImportError:
            print("Dask Distributed не знайдено. Встановіть за допомогою: pip install dask distributed")
            self.use_local = True
    
    def __del__(self):
        """Закриває клієнт при знищенні об'єкта."""
        if self.client:
            self.client.close()
    
    def parallel_map(self, func: Callable[[T], Any], items: List[T]) -> List[Any]:
        """
        Паралельно виконує функцію для кожного елемента списку.
        
        Args:
            func: Функція для виконання
            items: Список вхідних даних
            
        Returns:
            List[Any]: Список результатів
        """
        if self.use_local:
            # Для локального виконання просто використовуємо map
            return list(map(func, items))
        
        # Використовуємо Dask для паралельного виконання
        futures = self.client.map(func, items)
        return self.client.gather(futures)
    
    def parallel_optimize(
        self,
        algorithm_class: Union[type, str],
        algorithm_params: Dict[str, Any],
        initial_states: List[T],
        objective_function: Callable[[T], float],
        neighborhood_function: Callable[[T], List[T]],
        is_maximizing: bool = False,
        **kwargs
    ) -> Tuple[T, float, List[Tuple[int, float]]]:
        """
        Паралельно запускає алгоритм оптимізації з різними початковими станами.
        
        Args:
            algorithm_class: Клас алгоритму оптимізації або його назва для OptimizerFactory
            algorithm_params: Параметри алгоритму оптимізації
            initial_states: Список початкових станів
            objective_function: Функція оцінки стану
            neighborhood_function: Функція, що генерує сусідні стани
            is_maximizing: True якщо максимізуємо функцію, False якщо мінімізуємо
            **kwargs: Додаткові параметри для методу optimize
            
        Returns:
            Tuple[T, float, List[Tuple[int, float]]]: 
                Найкращий знайдений стан, його оцінка та історія оцінок
        """
        # Функція для виконання оптимізації з одним початковим станом
        def optimize_single(initial_state):
            # Створюємо екземпляр алгоритму
            if isinstance(algorithm_class, str):
                algorithm = OptimizerFactory.create_optimizer(algorithm_class, algorithm_params)
            else:
                algorithm = algorithm_class(**algorithm_params)
            
            # Запускаємо оптимізацію
            best_state, best_value, history = algorithm.optimize(
                initial_state=initial_state,
                objective_function=objective_function,
                neighborhood_function=neighborhood_function,
                is_maximizing=is_maximizing,
                **kwargs
            )
            
            return best_state, best_value, history
        
        # Виконуємо оптимізацію для кожного початкового стану
        start_time = time.time()
        
        if self.verbose:
            print(f"Запуск паралельної оптимізації з {len(initial_states)} початковими станами")
        
        # Паралельно запускаємо оптимізацію для кожного початкового стану
        results = self.parallel_map(optimize_single, initial_states)
        
        # Вибираємо найкращий результат
        if is_maximizing:
            best_idx = max(range(len(results)), key=lambda i: results[i][1])
        else:
            best_idx = min(range(len(results)), key=lambda i: results[i][1])
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Паралельна оптимізація завершена за {elapsed_time:.2f} секунд")
            print(f"Найкращий результат (з {len(results)} запусків): {results[best_idx][1]}")
        
        return results[best_idx]
    
    def read_csv(
        self, 
        file_path: str, 
        **kwargs
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Читає CSV-файл у Dask DataFrame або Pandas DataFrame.
        
        Args:
            file_path: Шлях до CSV-файлу
            **kwargs: Додаткові аргументи для pd.read_csv або dd.read_csv
            
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: Завантажений DataFrame
        """
        if self.use_local:
            return pd.read_csv(file_path, **kwargs)
        
        return dd.read_csv(file_path, **kwargs)
    
    def read_parquet(
        self, 
        file_path: str, 
        **kwargs
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Читає Parquet-файл у Dask DataFrame або Pandas DataFrame.
        
        Args:
            file_path: Шлях до Parquet-файлу
            **kwargs: Додаткові аргументи для pd.read_parquet або dd.read_parquet
            
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: Завантажений DataFrame
        """
        if self.use_local:
            return pd.read_parquet(file_path, **kwargs)
        
        return dd.read_parquet(file_path, **kwargs)
    
    def apply(
        self, 
        df: Union[pd.DataFrame, dd.DataFrame], 
        func: Callable, 
        axis: int = 1,
        meta: Any = None,
        **kwargs
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Застосовує функцію до кожного рядка або стовпця DataFrame.
        
        Args:
            df: DataFrame для обробки
            func: Функція для застосування
            axis: 0 для застосування до стовпців, 1 для застосування до рядків
            meta: Метадані для результату (для Dask)
            **kwargs: Додаткові аргументи для функції apply
            
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: Результат застосування функції
        """
        if self.use_local or isinstance(df, pd.DataFrame):
            return df.apply(func, axis=axis, **kwargs)
        
        # Для Dask потрібно вказати meta для визначення схеми результату
        return df.apply(func, axis=axis, meta=meta, **kwargs)
    
    def map_partitions(
        self, 
        df: dd.DataFrame, 
        func: Callable,
        meta: Any = None
    ) -> dd.DataFrame:
        """
        Застосовує функцію до кожної партиції Dask DataFrame.
        
        Args:
            df: Dask DataFrame для обробки
            func: Функція для застосування до кожної партиції
            meta: Метадані для результату
            
        Returns:
            dd.DataFrame: Результат застосування функції
        """
        if self.use_local or isinstance(df, pd.DataFrame):
            # Для Pandas немає партицій, тому просто застосовуємо функцію
            return func(df)
        
        return df.map_partitions(func, meta=meta)
    
    def aggregate(
        self, 
        df: Union[pd.DataFrame, dd.DataFrame], 
        group_by: List[str], 
        agg_funcs: Dict[str, List[str]]
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Групує та агрегує дані.
        
        Args:
            df: DataFrame для обробки
            group_by: Список стовпців для групування
            agg_funcs: Словник агрегаційних функцій (ключ - стовпець, значення - список функцій)
            
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: Результат агрегації
        """
        if self.use_local or isinstance(df, pd.DataFrame):
            return df.groupby(group_by).agg(agg_funcs)
        
        return df.groupby(group_by).agg(agg_funcs).compute()
    
    def transform_logistics_data(
        self, 
        packages_df: Union[pd.DataFrame, dd.DataFrame],
        nodes_df: Union[pd.DataFrame, dd.DataFrame],
        vehicles_df: Union[pd.DataFrame, dd.DataFrame]
    ) -> Dict[str, Union[pd.DataFrame, dd.DataFrame]]:
        """
        Трансформує логістичні дані для аналізу.
        
        Args:
            packages_df: DataFrame з посилками
            nodes_df: DataFrame з вузлами
            vehicles_df: DataFrame з транспортними засобами
            
        Returns:
            Dict[str, Union[pd.DataFrame, dd.DataFrame]]: Трансформовані дані
        """
        result = {}
        
        # Приклад трансформації: об'єднання даних про посилки і вузли
        if not self.use_local and isinstance(packages_df, dd.DataFrame):
            # Для Dask використовуємо merge
            packages_with_nodes = dd.merge(
                packages_df, 
                nodes_df.rename(columns={'id': 'origin_id', 'name': 'origin_name'}),
                on='origin_id',
                how='left'
            )
            
            # Об'єднуємо з вузлами призначення
            packages_with_nodes = dd.merge(
                packages_with_nodes,
                nodes_df.rename(columns={'id': 'destination_id', 'name': 'destination_name'}),
                on='destination_id',
                how='left'
            )
            
            result['packages_with_nodes'] = packages_with_nodes
        else:
            # Для Pandas використовуємо merge
            packages_with_nodes = pd.merge(
                packages_df,
                nodes_df[['id', 'name']].rename(columns={'id': 'origin_id', 'name': 'origin_name'}),
                on='origin_id',
                how='left'
            )
            
            # Об'єднуємо з вузлами призначення
            packages_with_nodes = pd.merge(
                packages_with_nodes,
                nodes_df[['id', 'name']].rename(columns={'id': 'destination_id', 'name': 'destination_name'}),
                on='destination_id',
                how='left'
            )
            
            result['packages_with_nodes'] = packages_with_nodes
        
        return result
    
    def analyze_package_flow(
        self, 
        packages_df: Union[pd.DataFrame, dd.DataFrame],
        time_period: str = 'D'  # 'D' для дня, 'W' для тижня, 'M' для місяця
    ) -> pd.DataFrame:
        """
        Аналізує потік посилок за часовими періодами.
        
        Args:
            packages_df: DataFrame з посилками (повинен мати стовпець 'timestamp')
            time_period: Період для групування
            
        Returns:
            pd.DataFrame: Результати аналізу
        """
        # Конвертуємо timestamp у datetime, якщо потрібно
        if 'timestamp' in packages_df.columns and packages_df['timestamp'].dtype != 'datetime64[ns]':
            if self.use_local or isinstance(packages_df, pd.DataFrame):
                packages_df['timestamp'] = pd.to_datetime(packages_df['timestamp'])
            else:
                packages_df['timestamp'] = dd.to_datetime(packages_df['timestamp'])
        
        # Групуємо за часовими періодами
        if 'timestamp' in packages_df.columns:
            if self.use_local or isinstance(packages_df, pd.DataFrame):
                # Для Pandas
                packages_by_time = packages_df.set_index('timestamp').resample(time_period).agg({
                    'id': 'count',
                    'weight': ['sum', 'mean', 'max'],
                    'volume': ['sum', 'mean', 'max']
                })
                
                # Перейменовуємо стовпці для зручності
                packages_by_time.columns = [f"{col[0]}_{col[1]}" for col in packages_by_time.columns]
                packages_by_time = packages_by_time.rename(columns={'id_count': 'package_count'})
                
                return packages_by_time
            else:
                # Для Dask (зауважте, що resample в Dask працює інакше)
                # Ми спочатку створюємо нові стовпці з періодами
                def add_period_column(df):
                    df = df.copy()
                    df['period'] = df['timestamp'].dt.to_period(time_period)
                    return df
                
                packages_df = self.map_partitions(packages_df, add_period_column)
                
                # Групуємо за періодами
                agg_result = packages_df.groupby('period').agg({
                    'id': 'count',
                    'weight': ['sum', 'mean', 'max'],
                    'volume': ['sum', 'mean', 'max']
                }).compute()
                
                # Перейменовуємо стовпці
                agg_result.columns = [f"{col[0]}_{col[1]}" for col in agg_result.columns]
                agg_result = agg_result.rename(columns={'id_count': 'package_count'})
                
                return agg_result
        else:
            print("Error: DataFrame does not have 'timestamp' column")
            return pd.DataFrame()
    
    def analyze_vehicle_utilization(
        self, 
        routes_df: Union[pd.DataFrame, dd.DataFrame],
        vehicles_df: Union[pd.DataFrame, dd.DataFrame]
    ) -> pd.DataFrame:
        """
        Аналізує використання транспортних засобів.
        
        Args:
            routes_df: DataFrame з маршрутами
            vehicles_df: DataFrame з транспортними засобами
            
        Returns:
            pd.DataFrame: Аналіз використання транспортних засобів
        """
        # Об'єднуємо маршрути і транспортні засоби
        if self.use_local or isinstance(routes_df, pd.DataFrame):
            # Для Pandas
            merged_df = pd.merge(
                routes_df,
                vehicles_df,
                left_on='vehicle_id',
                right_on='id',
                how='left',
                suffixes=('_route', '_vehicle')
            )
            
            # Аналізуємо використання транспортних засобів за типами
            utilization_by_type = merged_df.groupby('vehicle_type').agg({
                'id_route': 'count',
                'total_distance': ['sum', 'mean'],
                'total_time': ['sum', 'mean'],
                'load_weight': ['mean', 'max'],
                'load_volume': ['mean', 'max']
            })
            
            # Перейменовуємо стовпці
            utilization_by_type.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in utilization_by_type.columns]
            utilization_by_type = utilization_by_type.rename(columns={'id_route_count': 'route_count'})
            
            return utilization_by_type
        else:
            # Для Dask
            merged_df = dd.merge(
                routes_df,
                vehicles_df,
                left_on='vehicle_id',
                right_on='id',
                how='left',
                suffixes=('_route', '_vehicle')
            )
            
            # Групуємо та агрегуємо
            utilization_by_type = merged_df.groupby('vehicle_type').agg({
                'id_route': 'count',
                'total_distance': ['sum', 'mean'],
                'total_time': ['sum', 'mean'],
                'load_weight': ['mean', 'max'],
                'load_volume': ['mean', 'max']
            }).compute()
            
            # Перейменовуємо стовпці
            utilization_by_type.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in utilization_by_type.columns]
            utilization_by_type = utilization_by_type.rename(columns={'id_route_count': 'route_count'})
            
            return utilization_by_type
    
    def process_large_dataset(
        self,
        file_path: str,
        processing_func: Callable,
        output_path: Optional[str] = None,
        chunksize: int = 100000,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Обробляє великий набір даних з CSV-файлу, використовуючи чанки.
        
        Args:
            file_path: Шлях до CSV-файлу
            processing_func: Функція для обробки кожного чанка
            output_path: Шлях для збереження результату (якщо None, повертає результат)
            chunksize: Розмір чанка для читання
            **kwargs: Додаткові аргументи для pd.read_csv
            
        Returns:
            Optional[pd.DataFrame]: Результат обробки, якщо output_path не вказано
        """
        if self.use_local:
            # Для локального виконання використовуємо Pandas з чанками
            chunk_results = []
            
            for chunk in pd.read_csv(file_path, chunksize=chunksize, **kwargs):
                result = processing_func(chunk)
                if result is not None:
                    chunk_results.append(result)
            
            if chunk_results:
                final_result = pd.concat(chunk_results)
                
                if output_path:
                    final_result.to_csv(output_path, index=False)
                    return None
                
                return final_result
            
            return None
        else:
            # Для Dask читаємо весь файл як Dask DataFrame
            ddf = dd.read_csv(file_path, **kwargs)
            
            # Обробляємо дані за допомогою Dask
            result = self.map_partitions(ddf, processing_func)
            
            if output_path:
                # Зберігаємо результат у файл
                if output_path.endswith('.csv'):
                    result.to_csv(output_path, single_file=True, index=False)
                elif output_path.endswith('.parquet'):
                    result.to_parquet(output_path, engine='pyarrow')
                else:
                    # За замовчуванням зберігаємо як Parquet
                    result.to_parquet(output_path, engine='pyarrow')
                
                return None
            
            # Обчислюємо і повертаємо результат
            return result.compute()
    
    def predict_package_flow(
        self,
        historical_data: pd.DataFrame,
        forecast_periods: int = 7,
        time_col: str = 'date',
        target_col: str = 'package_count'
    ) -> pd.DataFrame:
        """
        Прогнозує потік посилок на основі історичних даних.
        
        Args:
            historical_data: DataFrame з історичними даними
            forecast_periods: Кількість періодів для прогнозування
            time_col: Назва стовпця з часовою міткою
            target_col: Назва стовпця з цільовою змінною
            
        Returns:
            pd.DataFrame: Прогноз потоку посилок
        """
        try:
            # Імпортуємо statsmodels для прогнозування часових рядів
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # Переконуємося, що дані відсортовані за часом
            historical_data = historical_data.sort_values(time_col)
            
            # Встановлюємо індекс за часовою міткою
            if historical_data[time_col].dtype != 'datetime64[ns]':
                historical_data[time_col] = pd.to_datetime(historical_data[time_col])
            
            time_series = historical_data.set_index(time_col)[target_col]
            
            # Визначаємо сезонність
            # Якщо дані щоденні, то сезонність може бути 7 (тижнева)
            # Якщо дані щомісячні, то сезонність може бути 12 (річна)
            if len(time_series) >= 365:  # Річні дані
                seasonal = 12
            elif len(time_series) >= 30:  # Місячні дані
                seasonal = 7
            else:
                seasonal = 1
            
            # Підбираємо SARIMA модель
            model = SARIMAX(
                time_series,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, seasonal)
            )
            
            model_fit = model.fit(disp=False)
            
            # Прогнозуємо майбутні значення
            forecast = model_fit.forecast(steps=forecast_periods)
            
            # Створюємо DataFrame з прогнозом
            last_date = time_series.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_periods,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                time_col: forecast_dates,
                target_col: forecast,
                'type': 'forecast'
            })
            
            # Додаємо історичні дані для порівняння
            historical_df = pd.DataFrame({
                time_col: time_series.index,
                target_col: time_series.values,
                'type': 'historical'
            })
            
            return pd.concat([historical_df, forecast_df]).reset_index(drop=True)
        
        except ImportError:
            print("Для прогнозування необхідно встановити statsmodels: pip install statsmodels")
            return pd.DataFrame()
        
        except Exception as e:
            print(f"Помилка при прогнозуванні: {str(e)}")
            return pd.DataFrame()