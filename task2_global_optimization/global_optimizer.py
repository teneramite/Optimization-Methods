"""
Глобальная оптимизация одномерной функции методом ломаных Пиявского

Алгоритм:
1. Оценка константы Липшица
2. Построение нижней огибающей функции
3. Поиск точки с максимальной характеристикой
4. Итеративное уточнение до достижения заданной точности
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import time
import warnings
warnings.filterwarnings('ignore')


class GlobalOptimizer:
    """Глобальная оптимизация методом ломаных Пиявского"""
    
    def __init__(self, func: Callable, a: float, b: float, epsilon: float = 0.01, r: float = 2.0):
        """
        Инициализация оптимизатора
        
        Args:
            func: целевая функция
            a: левая граница отрезка
            b: правая граница отрезка
            epsilon: точность (условие останова)
            r: параметр надёжности (r > 1, обычно 2-3)
        """
        self.func = func
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.r = r
        
        # История вычислений
        self.points = []  # точки, в которых вычислена функция
        self.values = []  # значения функции
        self.iterations_history = []  # история итераций
        
    def optimize(self, max_iterations: int = 1000) -> Tuple[float, float, dict]:
        """
        Поиск глобального минимума
        
        Args:
            max_iterations: максимальное количество итераций
            
        Returns:
            x_min: точка минимума
            f_min: значение функции в минимуме
            info: дополнительная информация
        """
        start_time = time.time()
        
        # Инициализация: вычисляем функцию на концах отрезка
        self.points = [self.a, self.b]
        self.values = [self.func(self.a), self.func(self.b)]
        
        print("=" * 80)
        print("ГЛОБАЛЬНАЯ ОПТИМИЗАЦИЯ МЕТОДОМ ПИЯВСКОГО")
        print("=" * 80)
        print(f"Отрезок: [{self.a}, {self.b}]")
        print(f"Точность: {self.epsilon}")
        print(f"Параметр надёжности r: {self.r}")
        print()
        
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Оценка константы Липшица
            L = self._estimate_lipschitz_constant()
            
            # Вычисление модифицированной константы
            m = self.r * L if L > 0 else 1.0
            
            # Выбор интервала с максимальной характеристикой
            max_char = -np.inf
            best_interval_idx = 0
            
            # Сортируем точки
            sorted_indices = np.argsort(self.points)
            sorted_points = [self.points[i] for i in sorted_indices]
            sorted_values = [self.values[i] for i in sorted_indices]
            
            # Для каждого интервала вычисляем характеристику
            for i in range(len(sorted_points) - 1):
                x_i = sorted_points[i]
                x_next = sorted_points[i + 1]
                f_i = sorted_values[i]
                f_next = sorted_values[i + 1]
                
                # Характеристика интервала
                delta = x_next - x_i
                char = m * delta + (f_i - f_next)**2 / (m * delta) - 2 * (f_i + f_next)
                
                if char > max_char:
                    max_char = char
                    best_interval_idx = i
            
            # Лучший интервал
            x_i = sorted_points[best_interval_idx]
            x_next = sorted_points[best_interval_idx + 1]
            f_i = sorted_values[best_interval_idx]
            f_next = sorted_values[best_interval_idx + 1]
            
            # Новая точка испытания (точка минимума огибающей на интервале)
            x_new = (x_i + x_next) / 2 - (f_next - f_i) / (2 * m)
            
            # Проверка, что точка внутри интервала
            x_new = max(x_i + self.epsilon / 10, min(x_next - self.epsilon / 10, x_new))
            
            # Вычисляем функцию в новой точке
            f_new = self.func(x_new)
            
            # Добавляем новую точку
            self.points.append(x_new)
            self.values.append(f_new)
            
            # Текущий минимум
            current_min_idx = np.argmin(self.values)
            current_min_x = self.points[current_min_idx]
            current_min_f = self.values[current_min_idx]
            
            # Сохраняем историю
            self.iterations_history.append({
                'iteration': iteration,
                'x_new': x_new,
                'f_new': f_new,
                'current_min_x': current_min_x,
                'current_min_f': current_min_f,
                'L': L,
                'm': m,
                'num_points': len(self.points)
            })
            
            if iteration % 10 == 0 or iteration <= 3:
                print(f"Итерация {iteration:3d}: "
                      f"точек={len(self.points):3d}, "
                      f"L={L:.4f}, "
                      f"x_min={current_min_x:.6f}, "
                      f"f_min={current_min_f:.6f}")
            
            # Условие останова: длина наибольшего интервала
            max_interval_length = max(sorted_points[i+1] - sorted_points[i] 
                                     for i in range(len(sorted_points) - 1))
            
            if max_interval_length < self.epsilon:
                print(f"\nДостигнута требуемая точность!")
                break
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Финальный результат
        min_idx = np.argmin(self.values)
        x_min = self.points[min_idx]
        f_min = self.values[min_idx]
        
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТ")
        print("=" * 80)
        print(f"Найденный минимум:")
        print(f"  x* = {x_min:.10f}")
        print(f"  f(x*) = {f_min:.10f}")
        print(f"Количество итераций: {iteration}")
        print(f"Количество вычислений функции: {len(self.points)}")
        print(f"Время выполнения: {elapsed_time:.4f} сек")
        print("=" * 80)
        
        info = {
            'iterations': iteration,
            'num_evaluations': len(self.points),
            'time': elapsed_time,
            'history': self.iterations_history
        }
        
        return x_min, f_min, info
    
    def _estimate_lipschitz_constant(self) -> float:
        """
        Оценка константы Липшица по имеющимся точкам
        
        Returns:
            L: оценка константы Липшица
        """
        if len(self.points) < 2:
            return 1.0
        
        max_slope = 0.0
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                x_i, x_j = self.points[i], self.points[j]
                f_i, f_j = self.values[i], self.values[j]
                
                if abs(x_j - x_i) > 1e-10:
                    slope = abs(f_j - f_i) / abs(x_j - x_i)
                    max_slope = max(max_slope, slope)
        
        return max_slope if max_slope > 0 else 1.0
    
    def visualize(self, save_path: str = None, show_iterations: List[int] = None):
        """
        Визуализация процесса оптимизации
        
        Args:
            save_path: путь для сохранения изображения
            show_iterations: список итераций для отображения
        """
        # Создаем сетку для отрисовки функции
        x_grid = np.linspace(self.a, self.b, 1000)
        y_grid = np.array([self.func(x) for x in x_grid])
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # График 1: Финальный результат
        ax1 = axes[0]
        ax1.plot(x_grid, y_grid, 'b-', linewidth=2, label='Функция f(x)')
        
        # Точки вычисления
        ax1.scatter(self.points, self.values, c='red', s=30, zorder=5, 
                   alpha=0.6, label='Точки вычисления')
        
        # Минимум
        min_idx = np.argmin(self.values)
        ax1.scatter(self.points[min_idx], self.values[min_idx], 
                   c='green', s=200, marker='*', zorder=10,
                   label=f'Минимум: ({self.points[min_idx]:.4f}, {self.values[min_idx]:.4f})')
        
        # Нижняя огибающая (финальная)
        self._plot_lower_envelope(ax1, alpha=0.3)
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('f(x)', fontsize=12)
        ax1.set_title('Результат глобальной оптимизации методом Пиявского', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # График 2: Сходимость
        ax2 = axes[1]
        
        if self.iterations_history:
            iterations = [h['iteration'] for h in self.iterations_history]
            min_values = [h['current_min_f'] for h in self.iterations_history]
            
            ax2.plot(iterations, min_values, 'b-', linewidth=2, marker='o', markersize=4)
            ax2.set_xlabel('Итерация', fontsize=12)
            ax2.set_ylabel('Текущий минимум f(x*)', fontsize=12)
            ax2.set_title('Сходимость алгоритма', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Финальное значение
            ax2.axhline(y=min_values[-1], color='r', linestyle='--', 
                       label=f'Финальный минимум: {min_values[-1]:.6f}')
            ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nВизуализация сохранена: {save_path}")
        
        plt.show()
    
    def _plot_lower_envelope(self, ax, alpha=0.3):
        """Отрисовка нижней огибающей"""
        # Оценка константы Липшица
        L = self._estimate_lipschitz_constant()
        m = self.r * L if L > 0 else 1.0
        
        # Сортируем точки
        sorted_indices = np.argsort(self.points)
        sorted_points = [self.points[i] for i in sorted_indices]
        sorted_values = [self.values[i] for i in sorted_indices]
        
        # Строим огибающую
        x_envelope = []
        y_envelope = []
        
        for i in range(len(sorted_points) - 1):
            x_i = sorted_points[i]
            x_next = sorted_points[i + 1]
            f_i = sorted_values[i]
            f_next = sorted_values[i + 1]
            
            # Точки на интервале
            x_interval = np.linspace(x_i, x_next, 50)
            
            # Огибающая: минимум из двух конусов
            for x in x_interval:
                y_left = f_i - m * abs(x - x_i)
                y_right = f_next - m * abs(x - x_next)
                y = max(y_left, y_right)
                
                x_envelope.append(x)
                y_envelope.append(y)
        
        if x_envelope:
            ax.plot(x_envelope, y_envelope, 'g--', linewidth=1.5, 
                   alpha=alpha, label='Нижняя огибающая')


def parse_function(func_str: str) -> Callable:
    """
    Парсинг строки функции
    
    Args:
        func_str: строка с функцией, например "x + sin(3.14159*x)"
        
    Returns:
        функция, принимающая x
    """
    import re
    from math import sin, cos, exp, log, sqrt, pi
    
    # Безопасное окружение для eval
    safe_dict = {
        'sin': sin, 'cos': cos, 'exp': exp, 'log': log, 'sqrt': sqrt,
        'pi': pi, 'abs': abs, '__builtins__': {}
    }
    
    def func(x):
        safe_dict['x'] = x
        try:
            return eval(func_str, safe_dict)
        except Exception as e:
            raise ValueError(f"Ошибка вычисления функции: {e}")
    
    return func


# Тестовые функции
def rastrigin_1d(x, A=10):
    """Одномерная функция Растригина"""
    return A + x**2 - A * np.cos(2 * np.pi * x)


def ackley_1d(x):
    """Одномерная функция Экли"""
    return -20 * np.exp(-0.2 * np.abs(x)) - np.exp(np.cos(2 * np.pi * x)) + 20 + np.e


def test_function_1(x):
    """Тестовая функция с несколькими локальными минимумами"""
    return (x - 2)**2 + 2 * np.sin(5 * x)


def test_function_2(x):
    """Сложная тестовая функция"""
    return x * np.sin(x) + 0.1 * x**2


if __name__ == '__main__':
    print("Демонстрация работы оптимизатора на функции Растригина\n")
    
    # Создаем оптимизатор
    optimizer = GlobalOptimizer(
        func=rastrigin_1d,
        a=-5.0,
        b=5.0,
        epsilon=0.01,
        r=2.0
    )
    
    # Запускаем оптимизацию
    x_min, f_min, info = optimizer.optimize(max_iterations=100)
    
    # Визуализация
    optimizer.visualize()
