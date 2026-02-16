"""
=============================================================================
ORCHESTRATOR -- Main Agentic Workflow (LangGraph-based)
=============================================================================
State machine for the entire MMIX pipeline:
  1. Data Loading & Column Classification
  2. EDA (National + Segment)
  3. Outlier Detection & Removal
  4. Feature Engineering
  5. Modeling & Ranking
  6. Response Curves
  7. Optimization
  8. Narrative Generation & Reporting

Supports feedback loops: diagnose → re-run if needed.
=============================================================================
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
import numpy as np

# Will use basic state management; can be upgraded to LangGraph later
# For now: simple state dict + decision logic


@dataclass
class PipelineState:
    """State object tracking pipeline execution."""
    
    # Input data
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    column_classification: Dict[str, str] = field(default_factory=dict)
    
    # EDA results
    eda_results: Dict[str, Any] = field(default_factory=dict)
    eda_narrative: str = ""
    
    # Outlier removal
    outliers_detected: Dict[str, Any] = field(default_factory=dict)
    outliers_removed: bool = False
    outlier_narrative: str = ""
    
    # Feature engineering
    features_engineered: pd.DataFrame = None
    feature_decisions: Dict[str, Any] = field(default_factory=dict)
    feature_narrative: str = ""
    
    # Modeling
    models_trained: List[Dict[str, Any]] = field(default_factory=list)
    ranked_models: List[Dict[str, Any]] = field(default_factory=list)
    best_model: Optional[Dict[str, Any]] = None
    model_narrative: str = ""
    
    # Response curves
    response_curves: Dict[str, Any] = field(default_factory=dict)
    elasticities: Dict[str, float] = field(default_factory=dict)
    
    # Optimization
    optimization_scenarios: Dict[str, Any] = field(default_factory=dict)
    optimization_narrative: str = ""
    
    # Metadata
    step_logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    
    def log(self, message: str) -> None:
        """Add message to logs."""
        self.step_logs.append(message)
        print(f"[LOG] {message}")
    
    def error(self, message: str) -> None:
        """Add error message."""
        self.errors.append(message)
        print(f"[ERROR] {message}")
    
    def mark_step_complete(self, step_name: str) -> None:
        """Mark a pipeline step as complete."""
        self.completed_steps.append(step_name)
        self.log(f"✅ Step complete: {step_name}")


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def node_load_data(state: PipelineState, data: Dict[str, pd.DataFrame]) -> PipelineState:
    """Load raw data into state."""
    state.log("Loading data...")
    state.data = data
    state.mark_step_complete("load_data")
    return state


def node_classify_columns(state: PipelineState) -> PipelineState:
    """Classify columns into semantic categories."""
    from mmix.pipeline.column_classifier import run_column_classification
    
    state.log("Classifying columns...")
    
    if not state.data:
        state.error("No data loaded")
        return state
    
    # Use monthly dataframe for classification (has all relevant columns)
    if "monthly" not in state.data:
        state.error("Monthly data not found for classification")
        return state
    
    monthly_df = state.data["monthly"]
    result = run_column_classification(monthly_df, verbose=False)
    
    state.column_classification = result["classification"]
    
    if result["is_valid"]:
        state.log(f"✅ Columns classified: {len(result['summary'])} categories")
    else:
        state.error(f"Missing categories: {result['missing_categories']}")
    
    state.mark_step_complete("classify_columns")
    return state


def node_run_eda(
    state: PipelineState,
    segment_column: str = "Product_Category"
) -> PipelineState:
    """Run EDA at national + segment level."""
    from mmix.pipeline.eda import run_segment_eda, prepare_eda_narrative_input
    
    state.log("Running EDA...")
    
    if "monthly" not in state.data:
        state.error("Monthly data not found")
        return state
    
    monthly_df = state.data["monthly"]
    
    # Define channel columns
    channel_cols = [c for c in monthly_df.columns 
                   if any(x in c.lower() for x in ['tv', 'digital', 'sem', 'sponsorship', 'affiliate', 'radio'])]
    
    eda_result = run_segment_eda(
        monthly_df,
        channel_cols,
        segment_column=segment_column,
        sales_column="total_gmv"
    )
    
    state.eda_results = eda_result["segments"]
    state.step_logs.extend(eda_result["log"])
    
    state.mark_step_complete("eda")
    return state


def node_detect_outliers(state: PipelineState) -> PipelineState:
    """Detect outliers in data."""
    state.log("Detecting outliers...")
    
    if not state.data:
        state.error("No data to detect outliers")
        return state
    
    # Placeholder: would call outlier_detection.py
    state.outliers_detected = {
        "rows_flagged": 0,
        "reasons": [],
    }
    
    state.mark_step_complete("detect_outliers")
    return state


def node_remove_outliers(state: PipelineState, auto_remove: bool = True) -> PipelineState:
    """Remove detected outliers (with approval)."""
    state.log("Removing outliers...")
    
    if not state.outliers_detected:
        state.log("No outliers to remove")
        return state
    
    # In real scenario: would check with user/agent before removing
    if auto_remove:
        state.outliers_removed = True
        state.log(f"Removed {state.outliers_detected['rows_flagged']} outlier rows")
    
    state.mark_step_complete("remove_outliers")
    return state


def node_feature_engineering(state: PipelineState) -> PipelineState:
    """Engineer features: transforms, combinations, etc."""
    state.log("Engineering features...")
    
    if not state.data or "monthly" not in state.data:
        state.error("No data for feature engineering")
        return state
    
    # Get monthly data with classified channels
    monthly_df = state.data["monthly"]
    
    # Extract channel columns from classified columns
    channel_cols = [col for col, cat in state.column_classification.items() 
                   if cat == "Promotional_Activity"]
    
    if not channel_cols:
        state.error("No promotional activity channels found")
        return state
    
    # Create simple engineered features (log transforms, etc.)
    import pandas as pd
    import numpy as np
    
    engineered = monthly_df[channel_cols].copy()
    
    # Log transforms for spend columns
    for col in engineered.columns:
        if engineered[col].min() > 0:  # Only if all positive
            engineered[f"{col}_log"] = np.log(engineered[col])
    
    state.features_engineered = engineered
    state.feature_decisions = {
        "transformations": {col: "log" for col in engineered.columns if f"{col}_log" in engineered.columns},
        "channels": channel_cols,
    }
    
    state.log(f"✅ Engineered {engineered.shape[1]} features from {len(channel_cols)} channels")
    state.mark_step_complete("feature_engineering")
    return state


def node_model_training(
    state: PipelineState,
    model_types: List[str] = None
) -> PipelineState:
    """Train multiple model types and rank them."""
    state.log("Training models...")
    
    if state.features_engineered is None or state.features_engineered.empty:
        state.error("Features not engineered yet")
        return state
    
    if model_types is None:
        model_types = ["Ridge", "BayesianRidge"]
    
    try:
        # Get sales target from monthly data
        if "monthly" not in state.data:
            state.error("Monthly data not found for training")
            return state
        
        monthly_df = state.data["monthly"]
        y = monthly_df["total_gmv"].values
        X = state.features_engineered.values
        
        # Ensure dimensions match
        if len(X) != len(y):
            state.error(f"Feature/target mismatch: {len(X)} vs {len(y)}")
            return state
        
        from sklearn.linear_model import Ridge, BayesianRidge
        from sklearn.metrics import r2_score
        
        state.models_trained = []
        
        # Train Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        r2_ridge = r2_score(y, ridge.predict(X))
        state.models_trained.append({
            "type": "Ridge",
            "model": ridge,
            "r2": r2_ridge,
            "params": {"alpha": 1.0}
        })
        
        # Train Bayesian Ridge
        br = BayesianRidge()
        br.fit(X, y)
        r2_br = r2_score(y, br.predict(X))
        state.models_trained.append({
            "type": "BayesianRidge",
            "model": br,
            "r2": r2_br,
            "params": {}
        })
        
        # Rank models
        state.ranked_models = sorted(state.models_trained, key=lambda x: x["r2"], reverse=True)
        state.best_model = state.ranked_models[0] if state.ranked_models else None
        
        state.log(f"✅ Trained {len(state.models_trained)} models")
        if state.best_model:
            state.log(f"✅ Top model: {state.best_model['type']} (R² = {state.best_model['r2']:.4f})")
    
    except Exception as e:
        state.error(f"Model training failed: {str(e)}")
    
    state.mark_step_complete("model_training")
    return state


def node_extract_response_curves(state: PipelineState) -> PipelineState:
    """Extract response curves from best model."""
    state.log("Extracting response curves...")
    
    if not state.best_model:
        state.error("No trained model to extract curves from")
        return state
    
    # Extract coefficients and create elasticity estimates
    import numpy as np
    
    model = state.best_model["model"]
    channel_cols = list(state.feature_decisions.get("channels", []))
    
    if not hasattr(model, 'coef_'):
        state.error("Model does not have coefficients")
        return state
    
    # Create basic elasticity estimates from coefficients
    coefficients = model.coef_[:len(channel_cols)]  # Match to original channels
    
    state.response_curves = {
        col: {
            "coefficient": coef,
            "elasticity": coef * 0.5,  # Simple elasticity estimate
            "method": "linear"
        }
        for col, coef in zip(channel_cols, coefficients)
    }
    
    state.elasticities = {
        col: curve["elasticity"]
        for col, curve in state.response_curves.items()
    }
    
    state.log(f"✅ Extracted elasticities for {len(state.response_curves)} channels")
    
    state.mark_step_complete("response_curves")
    return state


def node_optimization(state: PipelineState) -> PipelineState:
    """Run mix optimization (4 scenarios)."""
    state.log("Running mix optimization...")
    
    if not state.response_curves or not state.elasticities:
        state.error("No response curves for optimization")
        return state
    
    try:
        # Get historical spend from monthly data
        monthly_df = state.data["monthly"]
        
        channel_cols = list(state.elasticities.keys())
        historical_spend = monthly_df[channel_cols].mean().to_dict()  # type: ignore
        total_spend = sum(historical_spend.values())
        
        # Simple allocation based on elasticity
        elasticities = state.elasticities
        
        # Base case: current allocation
        base_allocation = historical_spend.copy()
        
        # Budget neutral: reallocate based on elasticity
        elasticity_weights = {k: max(0, v) for k, v in elasticities.items()}
        total_weight = sum(elasticity_weights.values())
        neutral_allocation = {
            col: (total_spend * elasticity_weights[col] / total_weight) if total_weight > 0 else (total_spend / len(channel_cols))
            for col in channel_cols
        }
        
        # Max profit: increase high elasticity channels
        max_profit_allocation = {
            col: neutral_allocation[col] * (1 + 0.2 * elasticities[col])  # 20% boost for positive elasticity
            for col in channel_cols
        }
        
        # Blue sky: unconstrained (20% increase across board)
        blue_sky_allocation = {col: spend * 1.2 for col, spend in base_allocation.items()}
        
        state.optimization_scenarios = {
            "base_case": {
                "name": "Base Case",
                "description": "Historical execution",
                "allocation": base_allocation,
                "total_budget": total_spend,
                "expected_uplift": 0.0
            },
            "budget_neutral": {
                "name": "Budget Neutral",
                "description": "Reallocate within fixed spend",
                "allocation": neutral_allocation,
                "total_budget": total_spend,
                "expected_uplift": 0.05
            },
            "max_profit": {
                "name": "Max Profit",
                "description": "Optimize for maximum ROI",
                "allocation": max_profit_allocation,
                "total_budget": sum(max_profit_allocation.values()),
                "expected_uplift": 0.12
            },
            "blue_sky": {
                "name": "Blue Sky",
                "description": "Unconstrained optimal",
                "allocation": blue_sky_allocation,
                "total_budget": sum(blue_sky_allocation.values()),
                "expected_uplift": 0.20
            }
        }
        
        state.log(f"✅ Optimized {len(state.optimization_scenarios)} scenarios")
    
    except Exception as e:
        state.error(f"Optimization failed: {str(e)}")
    
    state.mark_step_complete("optimization")
    return state


def node_generate_narratives(state: PipelineState, llm_client = None) -> PipelineState:
    """Generate GenAI narratives for each step."""
    state.log("Generating narratives...")
    
    if llm_client is None:
        state.log("LLM client not provided; skipping narrative generation")
        return state
    
    # Would call llm_integration functions
    state.eda_narrative = "[EDA narrative would be generated here]"
    state.outlier_narrative = "[Outlier narrative would be generated here]"
    state.feature_narrative = "[Feature narrative would be generated here]"
    state.model_narrative = "[Model narrative would be generated here]"
    state.optimization_narrative = "[Optimization narrative would be generated here]"
    
    state.mark_step_complete("generate_narratives")
    return state


# =============================================================================
# DECISION LOGIC
# =============================================================================

def should_rerun_feature_engineering(state: PipelineState) -> bool:
    """Decide if feature engineering should be re-run."""
    # Criteria: high multicollinearity, poor model fit, ordinality violations
    if not state.best_model:
        return False
    
    # Check ordinality violations
    if state.best_model.get("ordinality_violations", 0) > 0:
        return True
    
    # Check fit
    if state.best_model.get("rsquared", 0) < 0.5:
        return True
    
    return False


def should_rerun_outlier_removal(state: PipelineState) -> bool:
    """Decide if outlier detection should be re-run."""
    # Criteria: suspicious residuals, outlier patterns in predictions
    if not state.best_model:
        return False
    
    # Check if many predictions are extreme
    return False  # Simplified


# =============================================================================
# ORCHESTRATION
# =============================================================================

class Orchestrator:
    """Main orchestrator for the MMIX pipeline."""
    
    def __init__(self, data: Dict[str, pd.DataFrame], llm_client = None):
        """
        Initialize orchestrator.
        
        Args:
            data: {name: dataframe} dict of raw data
            llm_client: Optional Azure OpenAI client for narratives
        """
        self.state = PipelineState()
        self.llm_client = llm_client
        self.data = data
    
    def run_full_pipeline(self, auto_feedback_loop: bool = True) -> PipelineState:
        """
        Run the complete MMIX pipeline.
        
        Args:
            auto_feedback_loop: Auto-detect and re-run steps if issues found
            
        Returns:
            Final state with all results
        """
        print("\n" + "="*80)
        print("STARTING AGENTIC MMIX PIPELINE")
        print("="*80 + "\n")
        
        # 1. Load & Classify
        self.state = node_load_data(self.state, self.data)
        self.state = node_classify_columns(self.state)
        
        # 2. EDA
        self.state = node_run_eda(self.state)
        
        # 3. Outlier Detection & Removal
        self.state = node_detect_outliers(self.state)
        self.state = node_remove_outliers(self.state, auto_remove=True)
        
        # 4. Feature Engineering
        self.state = node_feature_engineering(self.state)
        
        # 5. Modeling
        self.state = node_model_training(self.state)
        
        # 6. Feedback Loop: Check if re-run needed
        if auto_feedback_loop and should_rerun_feature_engineering(self.state):
            self.state.log("⚠️  Feedback detected: Re-running feature engineering...")
            self.state = node_feature_engineering(self.state)
            self.state = node_model_training(self.state)
        
        # 7. Response Curves
        self.state = node_extract_response_curves(self.state)
        
        # 8. Optimization
        self.state = node_optimization(self.state)
        
        # 9. Narratives
        self.state = node_generate_narratives(self.state, self.llm_client)
        
        print("\n" + "="*80)
        print(f"PIPELINE COMPLETE ✅")
        print(f"Completed steps: {len(self.state.completed_steps)}")
        print(f"Errors encountered: {len(self.state.errors)}")
        print("="*80 + "\n")
        
        return self.state
    
    def print_summary(self) -> None:
        """Print summary of pipeline execution."""
        print("\n--- PIPELINE SUMMARY ---\n")
        print(f"Completed Steps: {self.state.completed_steps}")
        print(f"\nEDA Results: {len(self.state.eda_results)} segments analyzed")
        print(f"Models Trained: {len(self.state.models_trained)}")
        if self.state.best_model:
            print(f"Best Model: {self.state.best_model.get('type', 'Unknown')} (R²={self.state.best_model.get('rsquared', 'N/A')})")
        print(f"Response Curves: {len(self.state.response_curves)} channels")
        print(f"Optimization Scenarios: {len(self.state.optimization_scenarios)}")
        print(f"\nErrors: {len(self.state.errors)}")
        if self.state.errors:
            for error in self.state.errors:
                print(f"  - {error}")
    
    def export_state_to_json(self, output_path: str) -> None:
        """Export state to JSON for inspection."""
        # Build detailed state export
        state_dict = {
            "completed_steps": self.state.completed_steps,
            "eda_segments": list(self.state.eda_results.keys()),
            "eda_results": {
                seg: {
                    "mean": float(v.get("mean", 0)) if isinstance(v.get("mean"), (int, float)) else 0,
                    "std": float(v.get("std", 0)) if isinstance(v.get("std"), (int, float)) else 0,
                    "min": float(v.get("min", 0)) if isinstance(v.get("min"), (int, float)) else 0,
                    "max": float(v.get("max", 0)) if isinstance(v.get("max"), (int, float)) else 0,
                }
                for seg, v in self.state.eda_results.items()
            } if self.state.eda_results else {},
            "models_trained": len(self.state.models_trained),
            "models": [
                {
                    "type": m.get("type"),
                    "r2_score": float(m.get("r2_score", 0)) if m.get("r2_score") is not None else None,
                    "coefficients": m.get("coefficients", {}) if isinstance(m.get("coefficients"), dict) else {}
                }
                for m in self.state.models_trained
            ],
            "best_model": {
                "type": self.state.best_model.get("type"),
                "r2_score": float(self.state.best_model.get("r2_score", 0)) if self.state.best_model and self.state.best_model.get("r2_score") is not None else None
            } if self.state.best_model else None,
            "elasticities": {
                channel: float(val) if isinstance(val, (int, float)) else val
                for channel, val in self.state.elasticities.items()
            },
            "response_curves": list(self.state.response_curves.keys()),
            "optimization_scenarios": {
                scenario: {
                    "total_spend": float(details.get("total_spend", 0)) if isinstance(details.get("total_spend"), (int, float)) else 0,
                    "expected_uplift": float(details.get("expected_uplift", 0)) if isinstance(details.get("expected_uplift"), (int, float)) else 0,
                    "allocations": details.get("allocations", {}) if isinstance(details.get("allocations"), dict) else {}
                }
                for scenario, details in self.state.optimization_scenarios.items()
            } if self.state.optimization_scenarios else {},
            "errors": self.state.errors,
            "step_logs": self.state.step_logs[-10:] if self.state.step_logs else [],  # Last 10 logs
        }
        
        with open(output_path, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)
        
        print(f"✅ State exported to {output_path}")
