"""
================================================================================
EXPERIMENT 8 — ADAPTATION ALLOCATION ALGORITHMS
================================================================================
Enumerates the 27 possible action combinations for M=3 models (RF, ET, GB)
and implements budget allocation strategies:
  1. Frozen Allocation
  2. Continuous Allocation
  3. Baseline Event-Driven Allocation
  4. Random Selective Allocation
  5. Weakest-Model Selective Allocation
  6. Equal-Budget Selective Allocation
  7. Oracle Allocation
  8. Proposed Value-Based Allocation (Budget-Constrained & Lagrangian)
================================================================================
"""

import itertools
import numpy as np

ACTIONS = ['KEEP', 'PARTIAL', 'FULL']
COMPONENT_NAMES = ['RF', 'ET', 'GB']

ACTION_COSTS = {
    'KEEP': 0.0,
    'PARTIAL': 0.35,
    'FULL': 1.0,
}


def get_all_action_combinations():
    """Generates all 3^3 = 27 action combinations for (RF, ET, GB)."""
    return list(itertools.product(ACTIONS, repeat=3))


def compute_combination_cost(action_tuple):
    """Compute total computational cost for an action tuple (a_RF, a_ET, a_GB)."""
    return sum(ACTION_COSTS[a] for a in action_tuple)


def select_proposed_value_based(gain_estimates, budget=1.35, mode='budget_constrained', lagrangian_lambda=0.05):
    """
    Proposed Value-Based Allocation:
      - gain_estimates: dict mapping model_key -> dict of action -> estimated_gain
      - mode 'budget_constrained': max sum(G_i(a_i)) s.t. sum(C_i(a_i)) <= B
      - mode 'lagrangian': max sum(G_i(a_i) - lambda * C_i(a_i))
    """
    combinations = get_all_action_combinations()
    best_action = ('KEEP', 'KEEP', 'KEEP')
    best_val = -1e9

    for combo in combinations:
        c_rf, c_et, c_gb = combo
        total_cost = compute_combination_cost(combo)

        g_rf = gain_estimates['RF'].get(c_rf, 0.0)
        g_et = gain_estimates['ET'].get(c_et, 0.0)
        g_gb = gain_estimates['GB'].get(c_gb, 0.0)
        total_gain = g_rf + g_et + g_gb

        if mode == 'budget_constrained':
            if total_cost <= budget + 1e-6:
                if total_gain > best_val:
                    best_val = total_gain
                    best_action = combo
        elif mode == 'lagrangian':
            val = total_gain - lagrangian_lambda * total_cost
            if val > best_val:
                best_val = val
                best_action = combo

    return {
        'RF': best_action[0],
        'ET': best_action[1],
        'GB': best_action[2],
        'total_cost': compute_combination_cost(best_action),
        'estimated_gain': float(best_val),
    }


def select_oracle_allocation(realized_gains, budget=1.35):
    """
    Oracle Allocation:
    Selects optimal action combination based on actual realized gains post-adaptation.
    """
    combinations = get_all_action_combinations()
    best_action = ('KEEP', 'KEEP', 'KEEP')
    best_gain = -1e9

    for combo in combinations:
        c_rf, c_et, c_gb = combo
        total_cost = compute_combination_cost(combo)

        g_rf = realized_gains['RF'].get(c_rf, 0.0)
        g_et = realized_gains['ET'].get(c_et, 0.0)
        g_gb = realized_gains['GB'].get(c_gb, 0.0)
        total_gain = g_rf + g_et + g_gb

        if total_cost <= budget + 1e-6:
            if total_gain > best_gain:
                best_gain = total_gain
                best_action = combo

    return {
        'RF': best_action[0],
        'ET': best_action[1],
        'GB': best_action[2],
        'total_cost': compute_combination_cost(best_action),
        'realized_gain': float(best_gain),
    }


def select_random_selective(seed, budget=1.35):
    """
    Random Selective Adaptation:
    Uniformly samples an action combination satisfying cost <= budget.
    """
    rng = np.random.RandomState(seed)
    feasible = [c for c in get_all_action_combinations() if compute_combination_cost(c) <= budget + 1e-6]
    chosen = feasible[rng.choice(len(feasible))]
    return {
        'RF': chosen[0],
        'ET': chosen[1],
        'GB': chosen[2],
        'total_cost': compute_combination_cost(chosen),
    }


def select_weakest_selective(recent_model_f1s, budget=1.35):
    """
    Weakest-Model Adaptation:
    Prioritizes models with the lowest recent F1 performance.
    """
    # Sort models by recent F1 ascending
    sorted_models = sorted(recent_model_f1s.items(), key=lambda x: x[1])
    actions = {'RF': 'KEEP', 'ET': 'KEEP', 'GB': 'KEEP'}
    current_cost = 0.0

    # Try FULL for weakest, then PARTIAL
    for name, f1 in sorted_models:
        if current_cost + ACTION_COSTS['FULL'] <= budget + 1e-6:
            actions[name] = 'FULL'
            current_cost += ACTION_COSTS['FULL']
        elif current_cost + ACTION_COSTS['PARTIAL'] <= budget + 1e-6:
            actions[name] = 'PARTIAL'
            current_cost += ACTION_COSTS['PARTIAL']

    return {
        'RF': actions['RF'],
        'ET': actions['ET'],
        'GB': actions['GB'],
        'total_cost': current_cost,
    }


def select_equal_budget_selective(budget=1.35):
    """
    Equal-Budget Selective Adaptation:
    Distributes budget B / M equally among members.
    """
    per_model_budget = budget / 3.0
    if per_model_budget >= ACTION_COSTS['FULL']:
        action = 'FULL'
    elif per_model_budget >= ACTION_COSTS['PARTIAL']:
        action = 'PARTIAL'
    else:
        action = 'KEEP'

    actions = {'RF': action, 'ET': action, 'GB': action}
    return {
        'RF': action,
        'ET': action,
        'GB': action,
        'total_cost': compute_combination_cost((action, action, action)),
    }
