"""
Демонстрация работы решателя симплекс-методом
Вариант 17 + дополнительные примеры
"""

from simplex_solver import SimplexSolver, solve_from_file


def print_result(title, solution, value, status):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Статус: {status}")

    if solution is not None:
        print(f"Оптимальное значение: {value:.6f}")
        print("Оптимальная точка:")
        for i, x in enumerate(solution):
            print(f"  x{i+1} = {x:.6f}")
    else:
        print("Решение не найдено")
    print("=" * 80)

def demo_variant17():
    """Решение варианта 17"""
    solution, value, status = solve_from_file("input_variant17.json")
    print_result("ВАРИАНТ 17", solution, value, status)


def demo_simple_example():
    """Простой пример"""
    solver = SimplexSolver(
        c=[3, 2],
        A=[[2, 1], [2, 3], [3, 1]],
        b=[18, 42, 24],
        constraint_types=['<=', '<=', '<='],
        maximize=True
    )

    solution, value, status = solver.solve()
    print_result("Примерчик", solution, value, status)


def demo_with_equality_constraints():
    """Пример с равенствами"""
    solver = SimplexSolver(
        c=[2, 3],
        A=[[1, 1], [1, -1]],
        b=[4, 2],
        constraint_types=['=', '<='],
        maximize=True
    )

    solution, value, status = solver.solve()
    print_result("Пример с равенствами", solution, value, status)


if __name__ == "__main__":
    demo_variant17()
    demo_simple_example()
    demo_with_equality_constraints()
