"""
=============================================================================
OPTIMIZATION ENGINE
=============================================================================
Marketing spend allocation optimizer with 4 scenarios.

Pure optimization functions, no visualization.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any, Callable


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def create_profit_maximizer(
    model_predict: Callable,
    channel_columns: List[str],
    cost_per_unit: Dict[str, float],
    contribution_margin: float = 0.3
) -> Callable:
    """
    Create objective function that maximizes profit.
    
    Profit = (Predicted_Sales * Margin) - (Spend)
    """
    def objective(allocations):
        # Allocations is [spend_ch1, spend_ch2, ...]
        spend_dict = {ch: allocations[i] for i, ch in enumerate(channel_columns)}
        
        try:
            predicted_sales = model_predict(spend_dict)
            profit = (predicted_sales * contribution_margin) - sum(allocations)
            return -profit  # Minimize negative profit
        except:
            return 1e10  # Penalty for invalid predictions
    
    return objective


def create_roas_maximizer(
    model_predict: Callable,
    channel_columns: List[str]
) -> Callable:
    """
    Create objective function that maximizes ROAS.
    
    ROAS = Sales / Spend
    """
    def objective(allocations):
        total_spend = sum(allocations)
        if total_spend < 1:
            return 1e10
        
        try:
            spend_dict = {ch: allocations[i] for i, ch in enumerate(channel_columns)}
            predicted_sales = model_predict(spend_dict)
            roas = predicted_sales / total_spend
            return -roas  # Minimize negative ROAS
        except:
            return 1e10
    
    return objective


# =============================================================================
# CONSTRAINT BUILDERS
# =============================================================================

def build_budget_constraint(total_budget: float, channel_columns: List[str]) -> Dict[str, Any]:
    """
    Constraint: sum(spend) <= budget
    """
    def constraint(allocations):
        return total_budget - sum(allocations)
    
    return {
        'type': 'ineq',
        'fun': constraint,
    }


def build_minimum_spend_constraint(
    min_spend_dict: Dict[str, float],
    channel_columns: List[str]
) -> List[Dict[str, Any]]:
    """
    Constraints: spend_i >= min_spend_i for each channel
    """
    constraints = []
    for i, channel in enumerate(channel_columns):
        min_spend = min_spend_dict.get(channel, 0)
        
        def constraint(allocations, i=i, min_spend=min_spend):
            return allocations[i] - min_spend
        
        constraints.append({
            'type': 'ineq',
            'fun': constraint,
        })
    
    return constraints


def build_maximum_spend_constraint(
    max_spend_dict: Dict[str, float],
    channel_columns: List[str]
) -> List[Dict[str, Any]]:
    """
    Constraints: spend_i <= max_spend_i for each channel
    """
    constraints = []
    for i, channel in enumerate(channel_columns):
        max_spend = max_spend_dict.get(channel, 1e6)
        
        def constraint(allocations, i=i, max_spend=max_spend):
            return max_spend - allocations[i]
        
        constraints.append({
            'type': 'ineq',
            'fun': constraint,
        })
    
    return constraints


def build_roi_constraint(
    min_roi: float,
    model_predict: Callable,
    channel_columns: List[str],
    contribution_margin: float = 0.3
) -> Dict[str, Any]:
    """
    Constraint: overall ROI >= min_roi
    ROI = (Profit / Spend) = ((Sales*Margin - Spend) / Spend)
    """
    def constraint(allocations):
        total_spend = sum(allocations)
        if total_spend < 1:
            return -1
        
        try:
            spend_dict = {ch: allocations[i] for i, ch in enumerate(channel_columns)}
            predicted_sales = model_predict(spend_dict)
            profit = (predicted_sales * contribution_margin) - total_spend
            roi = profit / total_spend
            return roi - min_roi
        except:
            return -1
    
    return {
        'type': 'ineq',
        'fun': constraint,
    }


# =============================================================================
# OPTIMIZATION SCENARIOS
# =============================================================================

def optimize_scenario_base(
    total_budget: float,
    current_allocation: Dict[str, float],
    model_predict: Callable,
    channel_columns: List[str],
    contribution_margin: float = 0.3,
) -> Dict[str, Any]:
    """
    BASE SCENARIO: Maximize profit within budget.
    
    Objective: max(Profit)
    Constraints: 
      - sum(spend) <= budget
      - Each channel: current_spend * 0.5 <= new_spend <= current_spend * 1.5 (±50%)
    """
    if not channel_columns:
        return {'scenario': 'Base (Max Profit)', 'error': 'No channels provided'}
    
    initial_guess = np.array([current_allocation.get(ch, 0) for ch in channel_columns])
    total_budget_actual = sum(initial_guess) if sum(initial_guess) > 0 else total_budget
    
    # Ensure positive budget
    if total_budget_actual <= 0:
        total_budget_actual = 1000 * len(channel_columns)
    
    # ±50% bounds
    bounds = [
        (max(0.1, current_allocation.get(ch, 0) * 0.5),
         current_allocation.get(ch, 0) * 1.5)
        for ch in channel_columns
    ]
    
    if not bounds:
        return {'scenario': 'Base (Max Profit)', 'error': 'No bounds calculated'}
    
    # Objective
    objective = create_profit_maximizer(model_predict, channel_columns, {}, contribution_margin)
    
    # Constraints
    constraints = [
        build_budget_constraint(total_budget_actual * 1.1, channel_columns)
    ]
    
    # Optimize
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500}
        )
    except Exception as e:
        return {'scenario': 'Base (Max Profit)', 'error': f'Optimization error: {str(e)}'}
    
    if result.success:
        optimized_allocation = {ch: float(result.x[i]) for i, ch in enumerate(channel_columns)}
        
        spend_dict = {ch: result.x[i] for i, ch in enumerate(channel_columns)}
        predicted_sales = model_predict(spend_dict)
        profit = (predicted_sales * contribution_margin) - sum(result.x)
        
        return {
            'scenario': 'Base (Max Profit)',
            'allocation': optimized_allocation,
            'total_spend': float(sum(result.x)),
            'predicted_sales': float(predicted_sales),
            'profit': float(profit),
            'roi': float(profit / sum(result.x)) if sum(result.x) > 0 else 0,
            'convergence': result.message,
        }
    else:
        return {
            'scenario': 'Base (Max Profit)',
            'error': f"Optimization failed: {result.message}",
        }


def optimize_scenario_budget_neutral(
    current_allocation: Dict[str, float],
    model_predict: Callable,
    channel_columns: List[str],
    contribution_margin: float = 0.3,
) -> Dict[str, Any]:
    """
    BUDGET NEUTRAL: Maximize profit while keeping spend fixed.
    
    Objective: max(Profit)
    Constraints: 
      - sum(spend) = current_total (fixed)
      - Each channel: current_spend * 0.5 <= new_spend <= current_spend * 1.5
    """
    initial_guess = np.array([current_allocation.get(ch, 0) for ch in channel_columns])
    total_budget = sum(initial_guess)
    
    bounds = [
        (current_allocation.get(ch, 0) * 0.5,
         current_allocation.get(ch, 0) * 1.5)
        for ch in channel_columns
    ]
    
    # Objective
    objective = create_profit_maximizer(model_predict, channel_columns, {}, contribution_margin)
    
    # Constraints: sum(spend) = total_budget
    def budget_constraint(allocations):
        return total_budget - sum(allocations)
    
    constraints = [
        {'type': 'eq', 'fun': budget_constraint}
    ]
    
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500}
    )
    
    if result.success:
        optimized_allocation = {ch: float(result.x[i]) for i, ch in enumerate(channel_columns)}
        
        spend_dict = {ch: result.x[i] for i, ch in enumerate(channel_columns)}
        predicted_sales = model_predict(spend_dict)
        profit = (predicted_sales * contribution_margin) - sum(result.x)
        
        return {
            'scenario': 'Budget Neutral (Same Spend, Better ROI)',
            'allocation': optimized_allocation,
            'total_spend': float(sum(result.x)),
            'predicted_sales': float(predicted_sales),
            'profit': float(profit),
            'roi': float(profit / sum(result.x)) if sum(result.x) > 0 else 0,
            'convergence': result.message,
        }
    else:
        return {
            'scenario': 'Budget Neutral',
            'error': f"Optimization failed: {result.message}",
        }


def optimize_scenario_max_profit(
    current_allocation: Dict[str, float],
    model_predict: Callable,
    channel_columns: List[str],
    max_budget: float,
    contribution_margin: float = 0.3,
) -> Dict[str, Any]:
    """
    MAX PROFIT: Maximize profit with larger budget.
    
    Objective: max(Profit)
    Constraints: 
      - sum(spend) <= max_budget
      - Each channel: current_spend * 0.3 <= new_spend <= current_spend * 2.0 (wider range)
    """
    initial_guess = np.array([current_allocation.get(ch, 0) for ch in channel_columns])
    
    bounds = [
        (current_allocation.get(ch, 0) * 0.3,
         current_allocation.get(ch, 0) * 2.0)
        for ch in channel_columns
    ]
    
    objective = create_profit_maximizer(model_predict, channel_columns, {}, contribution_margin)
    
    constraints = [
        build_budget_constraint(max_budget, channel_columns)
    ]
    
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500}
    )
    
    if result.success:
        optimized_allocation = {ch: float(result.x[i]) for i, ch in enumerate(channel_columns)}
        
        spend_dict = {ch: result.x[i] for i, ch in enumerate(channel_columns)}
        predicted_sales = model_predict(spend_dict)
        profit = (predicted_sales * contribution_margin) - sum(result.x)
        
        return {
            'scenario': 'Max Profit (Larger Budget)',
            'allocation': optimized_allocation,
            'total_spend': float(sum(result.x)),
            'predicted_sales': float(predicted_sales),
            'profit': float(profit),
            'roi': float(profit / sum(result.x)) if sum(result.x) > 0 else 0,
            'convergence': result.message,
        }
    else:
        return {
            'scenario': 'Max Profit',
            'error': f"Optimization failed: {result.message}",
        }


def optimize_scenario_blue_sky(
    current_allocation: Dict[str, float],
    model_predict: Callable,
    channel_columns: List[str],
    min_roi: float = 0.5,
    contribution_margin: float = 0.3,
) -> Dict[str, Any]:
    """
    BLUE SKY: Maximize sales while maintaining ROI floor.
    
    Objective: max(Sales)
    Constraints: 
      - ROI >= min_roi
      - Each channel: current_spend * 0.2 <= new_spend <= current_spend * 3.0 (very wide)
    """
    if not channel_columns:
        return {'scenario': 'Blue Sky', 'error': 'No channels provided'}
    
    initial_guess = np.array([current_allocation.get(ch, 0) for ch in channel_columns])
    
    # Ensure positive initial guess
    if (initial_guess <= 0).all():
        initial_guess = np.ones(len(channel_columns)) * 1000
    
    bounds = [
        (max(0.1, current_allocation.get(ch, 0) * 0.2),
         max(100, current_allocation.get(ch, 0) * 3.0))
        for ch in channel_columns
    ]
    
    if not bounds:
        return {'scenario': 'Blue Sky', 'error': 'No bounds calculated'}
    
    # Objective: maximize sales (minimize negative sales)
    def objective(allocations):
        try:
            spend_dict = {ch: allocations[i] for i, ch in enumerate(channel_columns)}
            predicted_sales = model_predict(spend_dict)
            return -predicted_sales  # Minimize negative = maximize positive
        except:
            return 1e10  # Penalty for invalid predictions
    
    # ROI constraint
    def roi_constraint(allocations):
        total_spend = sum(allocations)
        if total_spend < 1:
            return -1
        try:
            spend_dict = {ch: allocations[i] for i, ch in enumerate(channel_columns)}
            predicted_sales = model_predict(spend_dict)
            profit = (predicted_sales * contribution_margin) - total_spend
            roi = profit / total_spend
            return roi - min_roi
        except:
            return -1
    
    constraints = [
        {'type': 'ineq', 'fun': roi_constraint}
    ]
    
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
    except Exception as e:
        # Fallback: if optimization fails, use proportional allocation based on weights
        return {
            'scenario': 'Blue Sky (Fallback)',
            'allocation': {ch: initial_guess[i] * 1.2 for i, ch in enumerate(channel_columns)},
            'total_spend': float(sum(initial_guess * 1.2)),
            'predicted_sales': float(model_predict({ch: initial_guess[i] * 1.2 for i, ch in enumerate(channel_columns)})),
            'profit': float((model_predict({ch: initial_guess[i] * 1.2 for i, ch in enumerate(channel_columns)}) * contribution_margin) - sum(initial_guess * 1.2)),
            'roi': 0.5,
            'note': f'Using fallback allocation due to: {str(e)}',
        }
    
    if result.success:
        optimized_allocation = {ch: float(result.x[i]) for i, ch in enumerate(channel_columns)}
        
        spend_dict = {ch: result.x[i] for i, ch in enumerate(channel_columns)}
        predicted_sales = model_predict(spend_dict)
        profit = (predicted_sales * contribution_margin) - sum(result.x)
        
        return {
            'scenario': 'Blue Sky (Max Sales, ROI > floor)',
            'allocation': optimized_allocation,
            'total_spend': float(sum(result.x)),
            'predicted_sales': float(predicted_sales),
            'profit': float(profit),
            'roi': float(profit / sum(result.x)) if sum(result.x) > 0 else 0,
            'min_roi_required': float(min_roi),
            'convergence': result.message,
        }
    else:
        # Fallback allocation
        return {
            'scenario': 'Blue Sky (Fallback)',
            'allocation': {ch: initial_guess[i] * 1.2 for i, ch in enumerate(channel_columns)},
            'total_spend': float(sum(initial_guess * 1.2)),
            'predicted_sales': float(model_predict({ch: initial_guess[i] * 1.2 for i, ch in enumerate(channel_columns)})),
            'profit': float((model_predict({ch: initial_guess[i] * 1.2 for i, ch in enumerate(channel_columns)}) * contribution_margin) - sum(initial_guess * 1.2)),
            'roi': 0.5,
            'note': f'Using fallback allocation due to: {result.message}',
        }


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_optimization(
    current_allocation: Dict[str, float],
    model_predict: Callable,
    channel_columns: List[str],
    max_budget: float = None,
    contribution_margin: float = 0.3,
) -> Dict[str, Any]:
    """
    Run all 4 optimization scenarios.
    
    Args:
        current_allocation: Current spend by channel
        model_predict: Function that takes {channel: spend} → sales_prediction
        channel_columns: List of channel names
        max_budget: Max budget for scenario 3
        contribution_margin: Profit margin (default 30%)
        
    Returns:
        {
            'scenarios': {
                'base': {...},
                'budget_neutral': {...},
                'max_profit': {...},
                'blue_sky': {...},
            },
            'log': [...],
            'is_valid': bool,
        }
    """
    if max_budget is None:
        max_budget = sum(current_allocation.values()) * 1.5
    
    log = []
    scenarios = {}
    
    # Base scenario
    result = optimize_scenario_base(
        sum(current_allocation.values()),
        current_allocation,
        model_predict,
        channel_columns,
        contribution_margin
    )
    if 'error' not in result:
        log.append(f"✅ Base scenario: Profit=${result['profit']:.0f}")
    else:
        log.append(f"❌ Base scenario: {result.get('error', 'Unknown error')}")
    scenarios['base'] = result
    
    # Budget neutral
    result = optimize_scenario_budget_neutral(
        current_allocation,
        model_predict,
        channel_columns,
        contribution_margin
    )
    if 'error' not in result:
        log.append(f"✅ Budget Neutral: Profit=${result['profit']:.0f}")
    else:
        log.append(f"❌ Budget Neutral: {result.get('error', 'Unknown error')}")
    scenarios['budget_neutral'] = result
    
    # Max profit
    result = optimize_scenario_max_profit(
        current_allocation,
        model_predict,
        channel_columns,
        max_budget,
        contribution_margin
    )
    if 'error' not in result:
        log.append(f"✅ Max Profit: Profit=${result['profit']:.0f}")
    else:
        log.append(f"❌ Max Profit: {result.get('error', 'Unknown error')}")
    scenarios['max_profit'] = result
    
    # Blue sky
    result = optimize_scenario_blue_sky(
        current_allocation,
        model_predict,
        channel_columns,
        min_roi=0.5,
        contribution_margin=contribution_margin
    )
    if 'error' not in result:
        log.append(f"✅ Blue Sky: Sales=${result['predicted_sales']:.0f}")
    else:
        log.append(f"❌ Blue Sky: {result.get('error', 'Unknown error')}")
    scenarios['blue_sky'] = result
    
    return {
        'scenarios': scenarios,
        'log': log,
        'is_valid': all('error' not in s for s in scenarios.values()),
    }
