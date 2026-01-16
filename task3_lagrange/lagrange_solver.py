"""
Решение задач условной оптимизации методом множителей Лагранжа

Вариант 7:
Минимизировать f(x,y) = (x-2)² + (y+3)² при условиях:
- g₁(x,y) = 3x + 2y - 4 = 0
- g₂(x,y) = x - y - 1 = 0
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def solve_lagrange_variant7():
    """Решение варианта 7 методом множителей Лагранжа"""
    
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ УСЛОВНОЙ ОПТИМИЗАЦИИ")
    print("МЕТОД МНОЖИТЕЛЕЙ ЛАГРАНЖА")
    print("=" * 80)
    
    # Определяем символьные переменные
    x, y, lambda1, lambda2 = sp.symbols('x y lambda1 lambda2', real=True)
    
    # Целевая функция
    f = (x - 2)**2 + (y + 3)**2
    
    # Ограничения-равенства
    g1 = 3*x + 2*y - 4
    g2 = x - y - 1
    
    print("\nЗАДАЧА:")
    print(f"Минимизировать: f(x,y) = {f}")
    print("При ограничениях:")
    print(f"  g₁(x,y) = {g1} = 0")
    print(f"  g₂(x,y) = {g2} = 0")
    
    # Составляем функцию Лагранжа
    L = f + lambda1 * g1 + lambda2 * g2
    
    print(f"\nФункция Лагранжа:")
    print(f"L(x,y,λ₁,λ₂) = f + λ₁·g₁ + λ₂·g₂")
    print(f"L = {L}")
    
    # Находим частные производные
    print("\nЧАСТНЫЕ ПРОИЗВОДНЫЕ:")
    dL_dx = sp.diff(L, x)
    dL_dy = sp.diff(L, y)
    dL_dlambda1 = sp.diff(L, lambda1)
    dL_dlambda2 = sp.diff(L, lambda2)
    
    print(f"∂L/∂x = {dL_dx}")
    print(f"∂L/∂y = {dL_dy}")
    print(f"∂L/∂λ₁ = {dL_dlambda1}")
    print(f"∂L/∂λ₂ = {dL_dlambda2}")
    
    # Составляем систему уравнений
    equations = [
        sp.Eq(dL_dx, 0),
        sp.Eq(dL_dy, 0),
        sp.Eq(dL_dlambda1, 0),
        sp.Eq(dL_dlambda2, 0)
    ]
    
    print("\nСИСТЕМА УРАВНЕНИЙ:")
    for i, eq in enumerate(equations, 1):
        print(f"  {i}. {eq}")
    
    # Решаем систему
    print("\nРЕШЕНИЕ СИСТЕМЫ...")
    solutions = sp.solve(equations, [x, y, lambda1, lambda2])
    
    print(f"\nНайдено решений: {len(solutions) if isinstance(solutions, list) else 1}")
    
    if not isinstance(solutions, list):
        solutions = [solutions]
    
    # Анализируем каждое решение
    results = []
    for i, sol in enumerate(solutions, 1):
        print(f"\n--- Решение {i} ---")
        x_val = float(sol[x])
        y_val = float(sol[y])
        lambda1_val = float(sol[lambda1])
        lambda2_val = float(sol[lambda2])
        
        print(f"x* = {x_val:.6f}")
        print(f"y* = {y_val:.6f}")
        print(f"λ₁ = {lambda1_val:.6f}")
        print(f"λ₂ = {lambda2_val:.6f}")
        
        # Значение целевой функции
        f_val = float(f.subs([(x, x_val), (y, y_val)]))
        print(f"f(x*, y*) = {f_val:.6f}")
        
        # Проверка ограничений
        g1_val = float(g1.subs([(x, x_val), (y, y_val)]))
        g2_val = float(g2.subs([(x, x_val), (y, y_val)]))
        print(f"\nПроверка ограничений:")
        print(f"  g₁(x*, y*) = {g1_val:.10f} ≈ 0 ✓" if abs(g1_val) < 1e-6 else f"  g₁ = {g1_val}")
        print(f"  g₂(x*, y*) = {g2_val:.10f} ≈ 0 ✓" if abs(g2_val) < 1e-6 else f"  g₂ = {g2_val}")
        
        results.append({
            'x': x_val,
            'y': y_val,
            'lambda1': lambda1_val,
            'lambda2': lambda2_val,
            'f_value': f_val
        })
    
    # Проверка достаточных условий (критерий Сильвестра)
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ДОСТАТОЧНЫХ УСЛОВИЙ ЭКСТРЕМУМА")
    print("=" * 80)
    
    # Матрица Гессе функции Лагранжа
    hessian_L = sp.Matrix([
        [sp.diff(L, x, 2), sp.diff(L, x, y)],
        [sp.diff(L, y, x), sp.diff(L, y, 2)]
    ])
    
    print("\nМатрица Гессе функции Лагранжа:")
    print(hessian_L)
    
    # Матрица Гессе целевой функции
    hessian_f = sp.Matrix([
        [sp.diff(f, x, 2), sp.diff(f, x, y)],
        [sp.diff(f, y, x), sp.diff(f, y, 2)]
    ])
    
    print("\nМатрица Гессе целевой функции f:")
    print(hessian_f)
    
    # Проверяем для каждого решения
    for i, sol in enumerate(solutions, 1):
        print(f"\n--- Анализ решения {i} ---")
        x_val, y_val = float(sol[x]), float(sol[y])
        
        # Вычисляем определители
        H_f = hessian_f.subs([(x, x_val), (y, y_val)])
        det_H = H_f.det()
        trace_H = H_f.trace()
        
        print(f"Матрица Гессе в точке ({x_val:.4f}, {y_val:.4f}):")
        print(H_f)
        print(f"Определитель: {float(det_H):.6f}")
        print(f"След: {float(trace_H):.6f}")
        
        # Критерий Сильвестра
        if float(det_H) > 0 and float(H_f[0,0]) > 0:
            print("Заключение: Точка МИНИМУМА (det > 0, H₁₁ > 0)")
        elif float(det_H) > 0 and float(H_f[0,0]) < 0:
            print("Заключение: Точка МАКСИМУМА (det > 0, H₁₁ < 0)")
        elif float(det_H) < 0:
            print("Заключение: Седловая точка (det < 0)")
        else:
            print("Заключение: Требуется дополнительное исследование")
    
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 80)
    best_result = min(results, key=lambda r: r['f_value'])
    print(f"Минимум достигается в точке:")
    print(f"  x* = {best_result['x']:.6f}")
    print(f"  y* = {best_result['y']:.6f}")
    print(f"Минимальное значение функции:")
    print(f"  f(x*, y*) = {best_result['f_value']:.6f}")
    print("=" * 80)
    
    return results


def visualize_problem(results):
    """Визуализация задачи и решения"""
    
    fig = plt.figure(figsize=(16, 6))
    
    # График 1: Линии уровня и ограничения
    ax1 = fig.add_subplot(121)
    
    # Сетка для целевой функции
    x_range = np.linspace(-2, 6, 200)
    y_range = np.linspace(-7, 1, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X - 2)**2 + (Y + 3)**2
    
    # Линии уровня
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Ограничения
    x_line = np.linspace(-2, 6, 200)
    
    # g₁: 3x + 2y - 4 = 0 => y = (4 - 3x) / 2
    y_g1 = (4 - 3*x_line) / 2
    ax1.plot(x_line, y_g1, 'r-', linewidth=2, label='g₁: 3x + 2y - 4 = 0')
    
    # g₂: x - y - 1 = 0 => y = x - 1
    y_g2 = x_line - 1
    ax1.plot(x_line, y_g2, 'b-', linewidth=2, label='g₂: x - y - 1 = 0')
    
    # Оптимальная точка
    for result in results:
        ax1.plot(result['x'], result['y'], 'g*', markersize=20, 
                label=f"Минимум ({result['x']:.2f}, {result['y']:.2f})")
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Линии уровня f(x,y) и ограничения', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 6)
    ax1.set_ylim(-7, 1)
    
    # График 2: 3D поверхность
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Поверхность целевой функции
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    
    # Ограничения как линии на поверхности
    Z_g1 = (x_line - 2)**2 + (y_g1 + 3)**2
    ax2.plot(x_line, y_g1, Z_g1, 'r-', linewidth=3, label='g₁')
    
    Z_g2 = (x_line - 2)**2 + (y_g2 + 3)**2
    ax2.plot(x_line, y_g2, Z_g2, 'b-', linewidth=3, label='g₂')
    
    # Оптимальная точка
    for result in results:
        ax2.scatter([result['x']], [result['y']], [result['f_value']], 
                   c='green', s=200, marker='*', label='Минимум')
    
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_zlabel('f(x,y)', fontsize=11)
    ax2.set_title('3D визуализация', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('variant7_solution.png', dpi=300, bbox_inches='tight')
    print("\nВизуализация сохранена: variant7_solution.png")
    plt.show()


if __name__ == '__main__':
    # Решение задачи
    results = solve_lagrange_variant7()
    
    # Визуализация
    visualize_problem(results)
