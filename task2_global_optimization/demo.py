"""
Демонстрация работы глобального оптимизатора
"""

from global_optimizer import GlobalOptimizer, rastrigin_1d, ackley_1d, test_function_1, test_function_2, parse_function
import numpy as np
import matplotlib.pyplot as plt


def demo_rastrigin():
    """Демонстрация на функции Растригина"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 1: ФУНКЦИЯ РАСТРИГИНА")
    print("="*80)
    print("\nФункция: f(x) = 10 + x² - 10·cos(2πx)")
    print("Глобальный минимум: x* = 0, f(x*) = 0")
    print("Множество локальных минимумов на отрезке [-5, 5]")
    
    optimizer = GlobalOptimizer(
        func=rastrigin_1d,
        a=-5.0,
        b=5.0,
        epsilon=0.01,
        r=2.5
    )
    
    x_min, f_min, info = optimizer.optimize(max_iterations=150)
    
    # Визуализация
    optimizer.visualize(save_path='rastrigin_result.png')
    
    return x_min, f_min, info


def demo_custom_function():
    """Демонстрация на пользовательской функции"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 2: СЛОЖНАЯ ФУНКЦИЯ")
    print("="*80)
    print("\nФункция: f(x) = (x-2)² + 2·sin(5x)")
    print("Несколько локальных минимумов")
    
    optimizer = GlobalOptimizer(
        func=test_function_1,
        a=-2.0,
        b=6.0,
        epsilon=0.01,
        r=2.0
    )
    
    x_min, f_min, info = optimizer.optimize(max_iterations=100)
    
    # Визуализация
    optimizer.visualize(save_path='custom_function_result.png')
    
    return x_min, f_min, info


def demo_ackley():
    """Демонстрация на функции Экли"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 3: ФУНКЦИЯ ЭКЛИ")
    print("="*80)
    print("\nФункция Экли (одномерная версия)")
    print("Глобальный минимум: x* = 0, f(x*) ≈ 0")
    
    optimizer = GlobalOptimizer(
        func=ackley_1d,
        a=-5.0,
        b=5.0,
        epsilon=0.01,
        r=3.0
    )
    
    x_min, f_min, info = optimizer.optimize(max_iterations=150)
    
    # Визуализация
    optimizer.visualize(save_path='ackley_result.png')
    
    return x_min, f_min, info


def demo_string_function():
    """Демонстрация парсинга строки функции"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 4: ПАРСИНГ СТРОКИ ФУНКЦИИ")
    print("="*80)
    
    func_str = "x + sin(3.14159*x)"
    print(f"\nФункция из строки: f(x) = {func_str}")
    
    func = parse_function(func_str)
    
    optimizer = GlobalOptimizer(
        func=func,
        a=-10.0,
        b=10.0,
        epsilon=0.01,
        r=2.0
    )
    
    x_min, f_min, info = optimizer.optimize(max_iterations=100)
    
    # Визуализация
    optimizer.visualize(save_path='string_function_result.png')
    
    return x_min, f_min, info


def compare_results():
    """Сравнение результатов на разных функциях"""
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    functions = [
        ("Растригин", rastrigin_1d, -5, 5, 0, 0),
        ("Экли", ackley_1d, -5, 5, 0, 0),
        ("Тестовая 1", test_function_1, -2, 6, None, None),
        ("x·sin(x)", test_function_2, -10, 10, None, None)
    ]
    
    results = []
    
    for name, func, a, b, true_x, true_f in functions:
        print(f"\n--- {name} ---")
        optimizer = GlobalOptimizer(func, a, b, epsilon=0.01, r=2.5)
        x_min, f_min, info = optimizer.optimize(max_iterations=100)
        
        result = {
            'name': name,
            'x_min': x_min,
            'f_min': f_min,
            'iterations': info['iterations'],
            'evaluations': info['num_evaluations'],
            'time': info['time']
        }
        
        if true_x is not None and true_f is not None:
            result['error_x'] = abs(x_min - true_x)
            result['error_f'] = abs(f_min - true_f)
        
        results.append(result)
    
    # Вывод таблицы результатов
    print("\n" + "="*80)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("="*80)
    print(f"{'Функция':<15} {'x*':>12} {'f(x*)':>12} {'Итер.':>7} {'Вычисл.':>10} {'Время(с)':>10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['name']:<15} {r['x_min']:>12.6f} {r['f_min']:>12.6f} "
              f"{r['iterations']:>7d} {r['evaluations']:>10d} {r['time']:>10.4f}")
    
    print("="*80)
    
    return results


def plot_all_functions():
    """Визуализация всех тестовых функций"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    functions = [
        ("Растригин", rastrigin_1d, -5, 5),
        ("Экли", ackley_1d, -5, 5),
        ("(x-2)² + 2sin(5x)", test_function_1, -2, 6),
        ("x·sin(x) + 0.1x²", test_function_2, -10, 10)
    ]
    
    for idx, (name, func, a, b) in enumerate(functions):
        ax = axes[idx // 2, idx % 2]
        x = np.linspace(a, b, 1000)
        y = np.array([func(xi) for xi in x])
        
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('f(x)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Отмечаем минимум
        min_idx = np.argmin(y)
        ax.scatter(x[min_idx], y[min_idx], c='red', s=100, marker='*', zorder=10)
    
    plt.tight_layout()
    plt.savefig('all_test_functions.png', dpi=300, bbox_inches='tight')
    print("График всех тестовых функций сохранен: all_test_functions.png")
    plt.show()


if __name__ == '__main__':
    # Визуализация всех тестовых функций
    plot_all_functions()
    
    # Демонстрация 1: Растригин
    demo_rastrigin()
    
    # Демонстрация 2: Пользовательская функция
    demo_custom_function()
    
    # Демонстрация 3: Экли
    demo_ackley()
    
    # Демонстрация 4: Парсинг строки
    demo_string_function()
    
    # Сравнение результатов
    compare_results()
    
    print("\n\nВсе демонстрации завершены!")
    print("Результаты сохранены в PNG файлах в текущей директории.")
