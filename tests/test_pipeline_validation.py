"""
=============================================================================
PIPELINE OUTPUT VALIDATION TESTS
=============================================================================
Validates that pipeline outputs are correct and sensible, not just error-free.

Checks:
  - Data shapes and types match expectations
  - Metrics are within reasonable bounds
  - Model scores make sense (R² between 0-1, etc.)
  - Optimization allocations sum correctly
  - No NaN or Inf values in outputs
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import PipelineState, run_pipeline
from engine import column_classification, eda_metrics, modeling, response_curves


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_eda_output(eda_results: Dict[str, Any]) -> Dict[str, bool]:
    """Validate EDA output structure and values."""
    checks = {}
    
    # Check structure (lowercase 'national')
    checks["has_national"] = "national" in eda_results
    if checks["has_national"]:
        nat = eda_results["national"]
        checks["has_rfe"] = "rfe" in nat
        checks["has_correlations"] = "correlations" in nat
        
        # Check RFE values are reasonable
        if "rfe" in nat:
            rfe = nat["rfe"]
            checks["rfe_reach_positive"] = rfe.get("reach", 0) >= 0
            checks["rfe_frequency_positive"] = rfe.get("frequency", 0) >= 0
            checks["rfe_engagement_valid"] = -1 <= rfe.get("engagement", 0) <= 1  # Correlation
    
    return checks


def validate_modeling_output(ranked_models: list, best_model: Dict[str, Any]) -> Dict[str, bool]:
    """Validate modeling output."""
    checks = {}
    
    checks["has_ranked_models"] = len(ranked_models) > 0
    checks["has_best_model"] = best_model is not None
    
    if best_model:
        # R² should be between 0 and 1
        r2 = best_model.get("r2", 0)
        checks["r2_valid"] = 0 <= r2 <= 1
        checks["r2_not_nan"] = not np.isnan(r2)
        
        # RMSE should be non-negative
        rmse = best_model.get("rmse", 0)
        checks["rmse_non_negative"] = rmse >= 0
        checks["rmse_not_nan"] = not np.isnan(rmse)
        
        # CV mean should be valid
        cv_mean = best_model.get("cv_mean", 0)
        checks["cv_mean_valid"] = 0 <= cv_mean <= 1
        
        # Model type should exist
        checks["model_type_exists"] = best_model.get("model_type") is not None
        
        # Feature count should be positive
        checks["feature_count_positive"] = best_model.get("feature_count", 0) > 0
    
    return checks


def validate_response_curves_output(curves: Dict[str, Any]) -> Dict[str, bool]:
    """Validate response curves output."""
    checks = {}
    
    checks["has_curves"] = len(curves) > 0
    
    for channel, curve_data in curves.items():
        # Each curve should have basic fields
        checks[f"{channel}_has_type"] = "curve_type" in curve_data
        checks[f"{channel}_has_elasticity"] = "elasticity" in curve_data
        checks[f"{channel}_has_r2"] = "r2" in curve_data
        
        # Elasticity should be valid number
        elasticity = curve_data.get("elasticity", 0)
        checks[f"{channel}_elasticity_valid"] = isinstance(elasticity, (int, float)) and not np.isnan(elasticity)
        
        # R² should be 0-1
        r2 = curve_data.get("r2", 0)
        checks[f"{channel}_r2_valid"] = 0 <= r2 <= 1
    
    return checks


def validate_optimization_output(scenarios: Dict[str, Any]) -> Dict[str, bool]:
    """Validate optimization scenarios output."""
    checks = {}
    
    checks["has_scenarios"] = len(scenarios) > 0
    
    expected_scenarios = ["base", "budget_neutral", "max_profit", "blue_sky"]
    for scenario_name in expected_scenarios:
        checks[f"has_{scenario_name}"] = scenario_name in scenarios
        
        if scenario_name in scenarios:
            scenario = scenarios[scenario_name]
            
            # Scenario is valid if it doesn't have 'error' field (fallback is OK)
            checks[f"{scenario_name}_valid"] = "error" not in scenario
            
            # Check allocation sums correctly (if allocation exists)
            allocation = scenario.get("allocation", {})
            total_spend = scenario.get("total_spend", 0)
            
            if allocation and total_spend > 0:
                allocation_sum = sum(allocation.values())
                # Allow 1% tolerance for rounding
                checks[f"{scenario_name}_allocation_sums"] = abs(allocation_sum - total_spend) < max(1, total_spend * 0.01)
            else:
                checks[f"{scenario_name}_allocation_sums"] = True  # OK if not applicable
            
            # Check profit/sales are reasonable
            profit = scenario.get("profit", 0)
            checks[f"{scenario_name}_profit_valid"] = not np.isnan(profit)
            
            sales = scenario.get("predicted_sales", 0)
            checks[f"{scenario_name}_sales_valid"] = sales >= 0 and not np.isnan(sales)
    
    return checks


def validate_column_classification(classification: Dict[str, str]) -> Dict[str, bool]:
    """Validate column classification."""
    checks = {}
    
    checks["has_classifications"] = len(classification) > 0
    
    # Should have at least some key categories
    categories = set(classification.values())
    checks["has_time_columns"] = "Time_Stamp" in categories
    checks["has_sales_columns"] = "Sales_Output" in categories
    checks["has_promotional_columns"] = "Promotional_Activity" in categories
    
    return checks


# =============================================================================
# FULL PIPELINE VALIDATION
# =============================================================================

def validate_full_pipeline(state: PipelineState) -> Dict[str, Dict[str, bool]]:
    """
    Validate entire pipeline output.
    
    Returns:
        Dict of {step_name: {check_name: bool}}
    """
    results = {}
    
    # Column classification
    if state.column_classification:
        results["column_classification"] = validate_column_classification(state.column_classification)
    
    # EDA
    if state.eda_results:
        results["eda"] = validate_eda_output(state.eda_results)
    
    # Modeling
    if state.ranked_models or state.best_model:
        results["modeling"] = validate_modeling_output(state.ranked_models, state.best_model)
    
    # Response curves
    if state.response_curves:
        results["response_curves"] = validate_response_curves_output(state.response_curves)
    
    # Optimization
    if state.optimization_scenarios:
        results["optimization"] = validate_optimization_output(state.optimization_scenarios)
    
    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_validation_report(validation_results: Dict[str, Dict[str, bool]]) -> bool:
    """
    Print validation report and return overall pass/fail.
    
    Returns:
        True if all checks pass, False otherwise
    """
    all_passed = True
    
    print("\n" + "=" * 80)
    print("PIPELINE OUTPUT VALIDATION REPORT")
    print("=" * 80)
    
    for step_name, checks in validation_results.items():
        print(f"\n{step_name.upper()}:")
        
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        
        print(f"  {passed}/{total} checks passed")
        
        # Show failures
        failures = {k: v for k, v in checks.items() if not v}
        if failures:
            all_passed = False
            print(f"  ❌ FAILURES:")
            for check_name, result in failures.items():
                print(f"    - {check_name}")
        else:
            print(f"  ✅ All checks passed")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ PIPELINE VALIDATION: ALL CHECKS PASSED")
    else:
        print("❌ PIPELINE VALIDATION: SOME CHECKS FAILED")
    print("=" * 80 + "\n")
    
    return all_passed


# =============================================================================
# TEST RUNNER
# =============================================================================

def test_pipeline_with_validation(data_path: str = "data/Secondfile.csv", mode: str = "deterministic") -> bool:
    """
    Run pipeline and validate all outputs.
    
    Args:
        data_path: Path to input CSV
        mode: "deterministic" or "agentic"
        
    Returns:
        True if pipeline runs and all validations pass
    """
    print(f"\n{'='*80}")
    print(f"RUNNING PIPELINE VALIDATION TEST")
    print(f"  Data: {data_path}")
    print(f"  Mode: {mode}")
    print(f"{'='*80}\n")
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load data: {str(e)}")
        return False
    
    # Initialize state
    state = PipelineState(
        data={"main": df},
        mode=mode,
        output_dir="test_output",
        verbose=False
    )
    
    # Run pipeline
    try:
        state = run_pipeline(state, llm_client=None)
        print(f"✅ Pipeline executed successfully")
    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate outputs
    validation_results = validate_full_pipeline(state)
    
    # Print report
    all_passed = print_validation_report(validation_results)
    
    return all_passed


if __name__ == "__main__":
    # Run test
    passed = test_pipeline_with_validation(
        data_path="data/Secondfile.csv",
        mode="deterministic"
    )
    
    sys.exit(0 if passed else 1)
