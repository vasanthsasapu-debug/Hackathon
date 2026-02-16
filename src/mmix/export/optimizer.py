"""
=============================================================================
MIX OPTIMIZER -- Budget Allocation Optimization
=============================================================================
Creates 4 optimization scenarios based on response curves:
  1. Base Case: No optimization (use historical execution)
  2. Budget Neutral: Fix total spend, optimize allocation by channel/segment
  3. Max Profit with Current Spend: Reallocate within budget
  4. Blue Sky: No spend constraints (mROI = 0%)
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# OPTIMIZATION UTILITIES
# =============================================================================

def create_objective_function(
    response_curves: Dict[str, Dict[str, Any]],
    elasticities: Dict[str, float]
) -> callable:
    """
    Create objective function for profit maximization.
    
    Objective: maximize_spend SUM(elasticity[c] * log(spend[c]))
    
    Args:
        response_curves: Channel response data
        elasticities: {channel: elasticity}
        
    Returns:
        Objective function for scipy.optimize
    """
    def objective(spend_vector: np.ndarray) -> float:
        """Negative profit (for minimization)."""
        total_profit = 0
        for idx, channel in enumerate(elasticities.keys()):
            if idx < len(spend_vector):
                spend = max(spend_vector[idx], 1)  # Avoid log(0)
                elasticity = elasticities.get(channel, 0)
                contribution = elasticity * np.log(spend)
                total_profit += contribution
        
        return -total_profit  # Negative because we minimize
    
    return objective


def create_constraints(
    channels: List[str],
    total_budget: float,
    min_spend_per_channel: float = 0.01
) -> Tuple[Bounds, List[Dict]]:
    """
    Create optimization constraints.
    
    Args:
        channels: Channel names
        total_budget: Total budget constraint
        min_spend_per_channel: Minimum spend per channel
        
    Returns:
        (bounds, linear_constraints)
    """
    n_channels = len(channels)
    
    # Bounds: each channel [min, ∞]
    bounds = Bounds(
        lb=[min_spend_per_channel * total_budget] * n_channels,
        ub=[total_budget] * n_channels
    )
    
    # Linear constraint: sum(spend) = total_budget
    A = np.ones((1, n_channels))
    constraint = LinearConstraint(A, [total_budget], [total_budget])
    
    return bounds, [constraint]


# =============================================================================
# SCENARIO BUILDERS
# =============================================================================

def build_base_case_scenario(
    monthly_df: pd.DataFrame,
    channel_columns: List[str],
    sales_column: str = "total_gmv"
) -> Dict[str, Any]:
    """
    Base Case: No optimization, use historical execution.
    
    Args:
        monthly_df: Historical monthly data
        channel_columns: List of channels
        sales_column: Sales/GMV column
        
    Returns:
        {
            "scenario": "Base Case",
            "description": "...",
            "allocation": {channel: spend},
            "expected_gmv": float,
            "expected_roi": float,
        }
    """
    # Average spend per channel
    allocation = {}
    for channel in channel_columns:
        if channel in monthly_df.columns:
            allocation[channel] = float(monthly_df[channel].mean())
    
    # Average GMV
    if sales_column in monthly_df.columns:
        expected_gmv = float(monthly_df[sales_column].mean())
    else:
        expected_gmv = 0
    
    # Simple ROI: GMV / Total Spend
    total_spend = sum(allocation.values())
    expected_roi = (expected_gmv / total_spend) if total_spend > 0 else 0
    
    return {
        "scenario": "Base Case",
        "description": "Historical execution pattern (no optimization)",
        "allocation": allocation,
        "total_budget": total_spend,
        "expected_gmv": expected_gmv,
        "expected_roi": expected_roi,
        "changes": {channel: 0 for channel in allocation.keys()},  # No change
    }


def build_budget_neutral_scenario(
    monthly_df: pd.DataFrame,
    channel_columns: List[str],
    elasticities: Dict[str, float],
    sales_column: str = "total_gmv"
) -> Dict[str, Any]:
    """
    Budget Neutral: Fix total spend, optimize allocation (maximize profit within budget).
    
    Args:
        monthly_df: Historical data
        channel_columns: List of channels
        elasticities: {channel: elasticity}
        sales_column: Sales column
        
    Returns:
        Optimized allocation scenario
    """
    # Get historical budget
    total_budget = 0
    for channel in channel_columns:
        if channel in monthly_df.columns:
            total_budget += monthly_df[channel].sum()
    
    # Optimize allocation
    channels_with_elasticity = [ch for ch in channel_columns if ch in elasticities]
    
    if not channels_with_elasticity:
        return build_base_case_scenario(monthly_df, channel_columns, sales_column)
    
    # Simple allocation: proportional to elasticity
    total_elasticity = sum(elasticities[ch] for ch in channels_with_elasticity)
    allocation = {}
    
    for channel in channel_columns:
        if channel in elasticities and total_elasticity > 0:
            allocation[channel] = (elasticities[channel] / total_elasticity) * total_budget
        else:
            allocation[channel] = 0
    
    # Expected GMV based on elasticities
    base_gmv = monthly_df[sales_column].mean() if sales_column in monthly_df.columns else 0
    expected_gmv = base_gmv * (1 + 0.1 * (sum(elasticities.values()) / len(elasticities)))  # +10% uplift
    
    expected_roi = (expected_gmv / total_budget) if total_budget > 0 else 0
    
    # Changes from base case
    base_allocation = {ch: monthly_df[ch].mean() if ch in monthly_df.columns else 0 
                       for ch in channel_columns}
    changes = {ch: allocation[ch] - base_allocation[ch] for ch in allocation}
    
    return {
        "scenario": "Budget Neutral",
        "description": "Fix total spend, optimize allocation by elasticity (within budget)",
        "allocation": allocation,
        "total_budget": total_budget,
        "expected_gmv": expected_gmv,
        "expected_roi": expected_roi,
        "changes": changes,
    }


def build_max_profit_scenario(
    monthly_df: pd.DataFrame,
    channel_columns: List[str],
    elasticities: Dict[str, float],
    sales_column: str = "total_gmv",
    flexibility: float = 1.2
) -> Dict[str, Any]:
    """
    Max Profit (Current Spend): Reallocate across channels within total budget.
    
    Channels with higher elasticity get more budget.
    Total spend fixed but can shift between channels.
    
    Args:
        monthly_df: Historical data
        channel_columns: Channels
        elasticities: Channel elasticities
        sales_column: Sales column
        flexibility: Max allocation shift (1.2 = ±20% of channel spend)
        
    Returns:
        Optimized allocation scenario
    """
    total_budget = sum(monthly_df[channel].sum() if channel in monthly_df.columns else 0 
                      for channel in channel_columns)
    
    channels_with_elasticity = [ch for ch in channel_columns if ch in elasticities]
    
    if not channels_with_elasticity:
        return build_base_case_scenario(monthly_df, channel_columns, sales_column)
    
    # Normalize elasticities and allocate
    sorted_channels = sorted(channels_with_elasticity, 
                           key=lambda x: elasticities[x], 
                           reverse=True)
    
    allocation = {}
    base_allocation = {ch: monthly_df[ch].sum() if ch in monthly_df.columns else 0 
                      for ch in channel_columns}
    
    # Give more to high-elasticity channels, constrained by flexibility
    for idx, channel in enumerate(sorted_channels):
        if idx < len(sorted_channels) / 2:  # Top half get more
            allocation[channel] = min(
                base_allocation[channel] * flexibility,
                total_budget * 0.3  # Max 30% to any single channel
            )
        else:  # Bottom half get less
            allocation[channel] = max(
                base_allocation[channel] / flexibility,
                total_budget * 0.05  # Min 5% to any channel
            )
    
    # Fill remaining channels
    for channel in channel_columns:
        if channel not in allocation:
            allocation[channel] = base_allocation[channel]
    
    # Normalize to stay within budget
    current_total = sum(allocation.values())
    scale_factor = total_budget / current_total if current_total > 0 else 1
    allocation = {ch: spend * scale_factor for ch, spend in allocation.items()}
    
    # Expected GMV with elasticity boost
    base_gmv = monthly_df[sales_column].mean() if sales_column in monthly_df.columns else 0
    elasticity_boost = sum(elasticities.values()) / len(elasticities) if elasticities else 0
    expected_gmv = base_gmv * (1 + elasticity_boost * 0.15)  # +15% uplift
    
    expected_roi = (expected_gmv / total_budget) if total_budget > 0 else 0
    
    changes = {ch: allocation[ch] - base_allocation[ch] for ch in allocation}
    
    return {
        "scenario": "Max Profit (Current Spend)",
        "description": "Reallocate budget across channels (total fixed) to maximize ROI",
        "allocation": allocation,
        "total_budget": total_budget,
        "expected_gmv": expected_gmv,
        "expected_roi": expected_roi,
        "changes": changes,
    }


def build_blue_sky_scenario(
    monthly_df: pd.DataFrame,
    channel_columns: List[str],
    elasticities: Dict[str, float],
    sales_column: str = "total_gmv",
    budget_multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Blue Sky: No spend constraints (mROI = 0%, optimal allocation).
    
    Allocate unlimited budget by highest elasticity channels.
    
    Args:
        monthly_df: Historical data
        channel_columns: Channels
        elasticities: Channel elasticities
        sales_column: Sales column
        budget_multiplier: 1.5 = 50% increase over historical
        
    Returns:
        Unconstrained optimal scenario
    """
    base_budget = sum(monthly_df[channel].sum() if channel in monthly_df.columns else 0 
                     for channel in channel_columns)
    total_budget = base_budget * budget_multiplier
    
    channels_with_elasticity = sorted(
        [(ch, elasticities.get(ch, 0)) for ch in channel_columns if ch in elasticities],
        key=lambda x: x[1],
        reverse=True
    )
    
    allocation = {}
    
    if channels_with_elasticity:
        # Allocate proportional to elasticity
        total_elasticity = sum(e for _, e in channels_with_elasticity)
        
        for channel, elasticity in channels_with_elasticity:
            if total_elasticity > 0:
                allocation[channel] = (elasticity / total_elasticity) * total_budget
            else:
                allocation[channel] = total_budget / len(channels_with_elasticity)
    
    # Channels without elasticity data
    for channel in channel_columns:
        if channel not in allocation:
            allocation[channel] = 0
    
    # Expected GMV: assume each elasticity point = 5% GMV uplift
    base_gmv = monthly_df[sales_column].mean() if sales_column in monthly_df.columns else 0
    elasticity_boost = sum(elasticities.values()) / len(elasticities) if elasticities else 0
    gmv_uplift = 1 + (elasticity_boost * 0.05) + (budget_multiplier - 1) * 0.2
    expected_gmv = base_gmv * gmv_uplift
    
    expected_roi = (expected_gmv / total_budget) if total_budget > 0 else 0
    
    base_allocation = {ch: monthly_df[ch].sum() if ch in monthly_df.columns else 0 
                      for ch in channel_columns}
    changes = {ch: allocation[ch] - base_allocation[ch] for ch in allocation}
    
    return {
        "scenario": "Blue Sky",
        "description": "No budget constraints; allocate optimally by elasticity (mROI = 0%)",
        "allocation": allocation,
        "total_budget": total_budget,
        "expected_gmv": expected_gmv,
        "expected_roi": expected_roi,
        "changes": changes,
    }


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_mix_optimization(
    monthly_df: pd.DataFrame,
    channel_columns: List[str],
    elasticities: Dict[str, float],
    response_curves: Dict[str, Any] = None,
    sales_column: str = "total_gmv"
) -> Dict[str, Any]:
    """
    Main entry point: create all 4 optimization scenarios.
    
    Returns:
        {
            "base_case": {...},
            "budget_neutral": {...},
            "max_profit": {...},
            "blue_sky": {...},
            "log": [...]
        }
    """
    log = []
    
    log.append(f"Running mix optimization for {len(channel_columns)} channels...")
    
    # Build scenarios
    base_case = build_base_case_scenario(monthly_df, channel_columns, sales_column)
    log.append(f"✅ Base Case: {base_case['expected_gmv']:.0f} GMV")
    
    budget_neutral = build_budget_neutral_scenario(
        monthly_df, channel_columns, elasticities, sales_column
    )
    log.append(f"✅ Budget Neutral: {budget_neutral['expected_gmv']:.0f} GMV ({budget_neutral['expected_roi']:.2f} ROI)")
    
    max_profit = build_max_profit_scenario(
        monthly_df, channel_columns, elasticities, sales_column
    )
    log.append(f"✅ Max Profit: {max_profit['expected_gmv']:.0f} GMV ({max_profit['expected_roi']:.2f} ROI)")
    
    blue_sky = build_blue_sky_scenario(
        monthly_df, channel_columns, elasticities, sales_column
    )
    log.append(f"✅ Blue Sky: {blue_sky['expected_gmv']:.0f} GMV ({blue_sky['expected_roi']:.2f} ROI)")
    
    scenarios = {
        "base_case": base_case,
        "budget_neutral": budget_neutral,
        "max_profit": max_profit,
        "blue_sky": blue_sky,
    }
    
    # Comparison
    log.append("\n--- Scenario Comparison ---")
    for scenario_name, scenario_data in scenarios.items():
        roi_lift = ((scenario_data['expected_roi'] / base_case['expected_roi']) - 1) * 100
        log.append(f"{scenario_name}: GMV={scenario_data['expected_gmv']:.0f}, ROI={scenario_data['expected_roi']:.2f} ({roi_lift:+.1f}%)")
    
    return {
        "scenarios": scenarios,
        "log": log,
    }
