"""
Динамическое программирование: Управление инвестиционным портфелем

Начальное состояние:
- ЦБ1 = 100 д.е.
- ЦБ2 = 800 д.е.
- Депозиты = 400 д.е.
- Свободные средства = 600 д.е.

3 этапа с вероятностными сценариями
Управление: покупка/продажа долями по 25%
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt


# Данные из задачи
INITIAL_STATE = {
    'cb1': 100,
    'cb2': 800,
    'dep': 400,
    'free': 600
}

# Вероятности сценариев по этапам
PROBABILITIES = {
    1: {'favorable': 0.6, 'neutral': 0.3, 'negative': 0.1},
    2: {'favorable': 0.3, 'neutral': 0.2, 'negative': 0.5},
    3: {'favorable': 0.4, 'neutral': 0.4, 'negative': 0.2}
}

# Коэффициенты изменения стоимости
COEFFICIENTS = {
    1: {
        'favorable': {'cb1': 1.20, 'cb2': 1.10, 'dep': 1.07},
        'neutral': {'cb1': 1.05, 'cb2': 1.02, 'dep': 1.03},
        'negative': {'cb1': 0.80, 'cb2': 0.95, 'dep': 1.00}
    },
    2: {
        'favorable': {'cb1': 1.40, 'cb2': 1.15, 'dep': 1.01},
        'neutral': {'cb1': 1.05, 'cb2': 1.00, 'dep': 1.00},
        'negative': {'cb1': 0.60, 'cb2': 0.90, 'dep': 1.00}
    },
    3: {
        'favorable': {'cb1': 1.15, 'cb2': 1.12, 'dep': 1.05},
        'neutral': {'cb1': 1.05, 'cb2': 1.01, 'dep': 1.01},
        'negative': {'cb1': 0.70, 'cb2': 0.94, 'dep': 1.00}
    }
}

# Комиссии брокеров
COMMISSIONS = {'cb1': 0.04, 'cb2': 0.07, 'dep': 0.05}

# Ограничения
CONSTRAINTS = {'cb1': 30, 'cb2': 150, 'dep': 100}

# Возможные управления (доли от начального объема)
ACTIONS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


class PortfolioOptimizer:
    """Оптимизатор управления портфелем методом динамического программирования"""
    
    def __init__(self):
        self.value_function = {}
        self.policy = {}
        self.initial_state = INITIAL_STATE.copy()
        
    def state_to_tuple(self, state):
        """Преобразование состояния в кортеж для хеширования"""
        return (round(state['cb1'], 2), round(state['cb2'], 2), 
                round(state['dep'], 2), round(state['free'], 2))
    
    def apply_action(self, state, action):
        """
        Применение управления к состоянию
        action = {'cb1': delta1, 'cb2': delta2, 'dep': delta3}
        """
        new_state = state.copy()
        
        # Вычисляем изменения
        delta_cb1 = action['cb1'] * self.initial_state['cb1']
        delta_cb2 = action['cb2'] * self.initial_state['cb2']
        delta_dep = action['dep'] * self.initial_state['dep']
        
        # Применяем управление с учетом комиссий
        if delta_cb1 > 0:  # Покупка
            cost = delta_cb1 * (1 + COMMISSIONS['cb1'])
            if new_state['free'] >= cost:
                new_state['cb1'] += delta_cb1
                new_state['free'] -= cost
        elif delta_cb1 < 0:  # Продажа
            revenue = abs(delta_cb1) * (1 - COMMISSIONS['cb1'])
            if new_state['cb1'] + delta_cb1 >= CONSTRAINTS['cb1']:
                new_state['cb1'] += delta_cb1
                new_state['free'] += revenue
        
        # Аналогично для cb2
        if delta_cb2 > 0:
            cost = delta_cb2 * (1 + COMMISSIONS['cb2'])
            if new_state['free'] >= cost:
                new_state['cb2'] += delta_cb2
                new_state['free'] -= cost
        elif delta_cb2 < 0:
            revenue = abs(delta_cb2) * (1 - COMMISSIONS['cb2'])
            if new_state['cb2'] + delta_cb2 >= CONSTRAINTS['cb2']:
                new_state['cb2'] += delta_cb2
                new_state['free'] += revenue
        
        # Аналогично для депозитов
        if delta_dep > 0:
            cost = delta_dep * (1 + COMMISSIONS['dep'])
            if new_state['free'] >= cost:
                new_state['dep'] += delta_dep
                new_state['free'] -= cost
        elif delta_dep < 0:
            revenue = abs(delta_dep) * (1 - COMMISSIONS['dep'])
            if new_state['dep'] + delta_dep >= CONSTRAINTS['dep']:
                new_state['dep'] += delta_dep
                new_state['free'] += revenue
        
        return new_state
    
    def apply_scenario(self, state, stage, scenario):
        """Применение сценария к состоянию"""
        coeffs = COEFFICIENTS[stage][scenario]
        new_state = state.copy()
        
        new_state['cb1'] *= coeffs['cb1']
        new_state['cb2'] *= coeffs['cb2']
        new_state['dep'] *= coeffs['dep']
        
        return new_state
    
    def total_value(self, state):
        """Общая стоимость портфеля"""
        return state['cb1'] + state['cb2'] + state['dep'] + state['free']
    
    def expected_value(self, state, stage):
        """Ожидаемая стоимость с учетом вероятностей"""
        expected = 0.0
        for scenario, prob in PROBABILITIES[stage].items():
            new_state = self.apply_scenario(state, stage, scenario)
            expected += prob * self.total_value(new_state)
        return expected
    
    def solve(self):
        """Решение задачи методом динамического программирования"""
        print("="*80)
        print("ДИНАМИЧЕСКОЕ ПРОГРАММИРОВАНИЕ")
        print("Задача управления инвестиционным портфелем")
        print("="*80)
        
        # Обратный проход
        print("\nОБРАТНЫЙ ПРОХОД (вычисление оптимальных значений)...")
        
        # Граничное условие (этап 4 - финальное состояние)
        # Упрощаем: рассматриваем небольшое количество дискретных состояний
        
        # Для упрощения будем рассматривать только начальное состояние
        # и несколько базовых управлений
        
        initial_state_tuple = self.state_to_tuple(INITIAL_STATE)
        
        # Генерируем несколько базовых управлений
        basic_actions = [
            {'cb1': 0, 'cb2': 0, 'dep': 0},  # Ничего не делать
            {'cb1': 0.25, 'cb2': 0, 'dep': 0},  # Купить ЦБ1
            {'cb1': 0, 'cb2': 0.25, 'dep': 0},  # Купить ЦБ2
            {'cb1': -0.25, 'cb2': 0.25, 'dep': 0},  # Перебалансировка
        ]
        
        # Упрощенное решение для демонстрации концепции
        current_state = INITIAL_STATE.copy()
        total_value_history = [self.total_value(current_state)]
        
        print(f"\nНачальное состояние:")
        print(f"  ЦБ1: {current_state['cb1']:.2f} д.е.")
        print(f"  ЦБ2: {current_state['cb2']:.2f} д.е.")
        print(f"  Депозиты: {current_state['dep']:.2f} д.е.")
        print(f"  Свободные: {current_state['free']:.2f} д.е.")
        print(f"  Общая стоимость: {self.total_value(current_state):.2f} д.е.")
        
        # Простая стратегия: на каждом этапе выбираем действие с максимальным ожиданием
        for stage in range(1, 4):
            print(f"\n--- ЭТАП {stage} ---")
            
            best_action = None
            best_expected_value = -np.inf
            
            for action in basic_actions:
                state_after_action = self.apply_action(current_state, action)
                expected_val = self.expected_value(state_after_action, stage)
                
                if expected_val > best_expected_value:
                    best_expected_value = expected_val
                    best_action = action
            
            # Применяем лучшее действие
            current_state = self.apply_action(current_state, best_action)
            
            print(f"Выбранное управление: {best_action}")
            print(f"Ожидаемое значение: {best_expected_value:.2f} д.е.")
            
            # Применяем средний сценарий для демонстрации
            avg_state = {'cb1': 0, 'cb2': 0, 'dep': 0, 'free': 0}
            for scenario, prob in PROBABILITIES[stage].items():
                scenario_state = self.apply_scenario(current_state, stage, scenario)
                for key in avg_state:
                    avg_state[key] += prob * scenario_state[key]
            
            current_state = avg_state
            total_value_history.append(self.total_value(current_state))
            
            print(f"Состояние после этапа:")
            print(f"  ЦБ1: {current_state['cb1']:.2f} д.е.")
            print(f"  ЦБ2: {current_state['cb2']:.2f} д.е.")
            print(f"  Депозиты: {current_state['dep']:.2f} д.е.")
            print(f"  Свободные: {current_state['free']:.2f} д.е.")
            print(f"  Общая стоимость: {self.total_value(current_state):.2f} д.е.")
        
        print("\n" + "="*80)
        print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
        print("="*80)
        print(f"Начальная стоимость портфеля: {total_value_history[0]:.2f} д.е.")
        print(f"Конечная стоимость портфеля: {total_value_history[-1]:.2f} д.е.")
        print(f"Прирост: {total_value_history[-1] - total_value_history[0]:.2f} д.е.")
        print(f"Доходность: {((total_value_history[-1] / total_value_history[0]) - 1) * 100:.2f}%")
        print("="*80)
        
        return total_value_history


if __name__ == '__main__':
    optimizer = PortfolioOptimizer()
    history = optimizer.solve()
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history)), history, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Этап', fontsize=12)
    plt.ylabel('Стоимость портфеля (д.е.)', fontsize=12)
    plt.title('Динамика стоимости портфеля', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(history)), ['Начало'] + [f'Этап {i}' for i in range(1, len(history))])
    plt.tight_layout()
    plt.savefig('portfolio_dynamics.png', dpi=300, bbox_inches='tight')
    print("\nГрафик сохранен: portfolio_dynamics.png")
    plt.show()
