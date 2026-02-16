"""
=============================================================================
ENHANCED MODELING -- Extended Models + Ranking
=============================================================================
Additions to existing modeling_engine.py:
  1. GLM models (Poisson, Gamma)
  2. Fixed Effects & Random Effects models
  3. Composite model scoring (Fit, Stability, Ordinality)
  4. Top-10 model ranking
  5. GenAI-ready output for model comparison
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (14, 6)
sns.set_style("whitegrid")


# =============================================================================
# GLM MODELS
# =============================================================================

def fit_glm_poisson(X: pd.DataFrame, y: pd.Series, formula: str = None) -> Dict[str, Any]:
    """
    Fit Poisson GLM (good for count data).
    
    Args:
        X: Feature matrix
        y: Target variable
        formula: Optional statsmodels formula
        
    Returns:
        {model, summary, coef, aic, bic}
    """
    try:
        X_with_const = sm.add_constant(X)
        model = sm.GLM(y, X_with_const, family=sm.families.Poisson())
        result = model.fit()
        
        return {
            "model": result,
            "type": "GLM_Poisson",
            "coef": result.params.to_dict(),
            "aic": result.aic,
            "bic": result.bic,
            "deviance": result.deviance,
            "rsquared": r2_score(y, result.predict(X_with_const)),
            "summary": str(result.summary()),
        }
    except Exception as e:
        return {"error": str(e), "type": "GLM_Poisson"}


def fit_glm_gamma(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Fit Gamma GLM (good for positive continuous data).
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        {model, summary, coef, aic, bic}
    """
    try:
        X_with_const = sm.add_constant(X)
        model = sm.GLM(y, X_with_const, family=sm.families.Gamma())
        result = model.fit()
        
        return {
            "model": result,
            "type": "GLM_Gamma",
            "coef": result.params.to_dict(),
            "aic": result.aic,
            "bic": result.bic,
            "deviance": result.deviance,
            "rsquared": r2_score(y, result.predict(X_with_const)),
            "summary": str(result.summary()),
        }
    except Exception as e:
        return {"error": str(e), "type": "GLM_Gamma"}


# =============================================================================
# FIXED EFFECTS & RANDOM EFFECTS
# =============================================================================

def fit_fixed_effects(
    df: pd.DataFrame,
    formula: str,
    group_column: str
) -> Dict[str, Any]:
    """
    Fit Fixed Effects model (within-group variation).
    
    Example: 'sales ~ tv + digital + sponsorship'
    Automatically demeaning by group to remove group-level intercepts.
    
    Args:
        df: DataFrame with formula variables + group_column
        formula: Standard regression formula
        group_column: Column name for grouping (segment, region, etc.)
        
    Returns:
        {model, coef, r_squared, summary}
    """
    try:
        # Fit with group dummies (equivalent to FE)
        model = sm.formula.ols(formula + f" + C({group_column})", data=df).fit()
        
        # Extract coefficients excluding group dummies
        coef_dict = {
            k: v for k, v in model.params.items() 
            if not k.startswith(f"C({group_column})")
        }
        
        return {
            "model": model,
            "type": "FixedEffects",
            "coef": coef_dict,
            "rsquared": model.rsquared,
            "adj_rsquared": model.rsquared_adj,
            "summary": str(model.summary()),
        }
    except Exception as e:
        return {"error": str(e), "type": "FixedEffects"}


def fit_mixed_effects(
    df: pd.DataFrame,
    formula: str,
    group_column: str
) -> Dict[str, Any]:
    """
    Fit Mixed Effects model (random intercepts by group).
    
    Args:
        df: DataFrame with formula variables + group_column
        formula: Standard regression formula
        group_column: Column for random intercepts
        
    Returns:
        {model, coef, aic, bic, summary}
    """
    try:
        # Use statsmodels MixedLM for random effects
        md = sm.formula.mixedlm(formula, df, groups=df[group_column])
        result = md.fit()
        
        return {
            "model": result,
            "type": "MixedEffects",
            "coef": result.fe_params.to_dict(),
            "aic": result.aic,
            "bic": result.bic,
            "summary": str(result.summary()),
        }
    except Exception as e:
        return {"error": str(e), "type": "MixedEffects"}


# =============================================================================
# MODEL COMPARISON & SCORING
# =============================================================================

def calculate_model_stability(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray
) -> Dict[str, float]:
    """
    Calculate model stability metrics.
    
    Stability = how consistent predictions are (low variance in residuals).
    
    Args:
        y_actual: Actual values
        y_pred: Predicted values
        residuals: Prediction residuals
        
    Returns:
        {
            "std_residuals": float,
            "mean_abs_error": float,
            "stability_score": float (0-1, higher is better)
        }
    """
    mae = np.mean(np.abs(residuals))
    std_res = np.std(residuals)
    
    # Stability inversely proportional to residual variance
    # Normalize: assume std_res < 0.5*mean(y) is "good"
    max_std = np.mean(np.abs(y_actual)) * 0.5
    stability_score = max(0, 1 - (std_res / max_std)) if max_std > 0 else 0.5
    
    return {
        "std_residuals": round(std_res, 4),
        "mean_abs_error": round(mae, 4),
        "stability_score": round(stability_score, 3),
    }


def check_ordinality(
    coef_dict: Dict[str, float],
    important_channels: List[str] = None
) -> Dict[str, Any]:
    """
    Check ordinality (monotonicity) for important channels.
    
    Important channels should have coefficient > 0 (positive elasticity).
    
    Args:
        coef_dict: Model coefficients
        important_channels: Channels that should be positive (e.g., "Calls", "TV")
        
    Returns:
        {
            "ordinality_violations": int,
            "violation_details": [...],
            "ordinality_score": float (0-1)
        }
    """
    if important_channels is None:
        important_channels = ["log_Total_Investment", "log_spend_traditional", "log_spend_digital_performance"]
    
    violations = []
    
    for channel in important_channels:
        if channel in coef_dict:
            coef = coef_dict[channel]
            if coef < 0:
                violations.append({
                    "channel": channel,
                    "coefficient": round(coef, 4),
                    "expected": "> 0",
                })
    
    # Ordinality score: 1 if no violations, 0.5 if some, 0 if all
    ordinality_score = max(0, 1 - (len(violations) / len(important_channels)))
    
    return {
        "ordinality_violations": len(violations),
        "violation_details": violations,
        "ordinality_score": round(ordinality_score, 3),
    }


def calculate_composite_score(
    model_results: Dict[str, Any],
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate composite model score based on multiple criteria.
    
    Score = w_fit * R² + w_stability * Stability + w_ordinality * Ordinality
    
    Args:
        model_results: Dict with keys like 'rsquared', 'ordinality_score'
        y_actual: Actual values
        y_pred: Predicted values
        weights: {fit, stability, ordinality} weights (default 0.5, 0.3, 0.2)
        
    Returns:
        float: Composite score (0-1)
    """
    if weights is None:
        weights = {"fit": 0.5, "stability": 0.3, "ordinality": 0.2}
    
    # Extract metrics
    fit_score = model_results.get("rsquared", 0)
    
    residuals = y_actual - y_pred
    stability = calculate_model_stability(y_actual, y_pred, residuals)["stability_score"]
    
    ordinality = model_results.get("ordinality_score", 0.5)
    
    # Composite
    composite = (
        weights["fit"] * fit_score +
        weights["stability"] * stability +
        weights["ordinality"] * ordinality
    )
    
    return round(composite, 3)


def rank_models(
    models_results: List[Dict[str, Any]],
    y_actual: np.ndarray,
    y_pred_list: List[np.ndarray],
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Rank models by composite score and return top N.
    
    Args:
        models_results: List of model result dicts
        y_actual: Actual values
        y_pred_list: List of predictions (same length as models_results)
        top_n: Return top N models
        
    Returns:
        Sorted list of top models with rank and score
    """
    ranked_models = []
    
    for idx, (model_res, y_pred) in enumerate(zip(models_results, y_pred_list)):
        if "error" in model_res:
            continue
        
        score = calculate_composite_score(model_res, y_actual, y_pred)
        
        ranked_models.append({
            "rank": None,  # Set below
            "model_id": model_res.get("type", f"Model_{idx}"),
            "composite_score": score,
            "fit_r2": round(model_res.get("rsquared", 0), 3),
            "ordinality": model_res.get("ordinality_score", 0),
            "model": model_res,
        })
    
    # Sort by composite score
    ranked_models.sort(key=lambda x: x["composite_score"], reverse=True)
    
    # Add ranks
    for idx, model in enumerate(ranked_models[:top_n]):
        model["rank"] = idx + 1
    
    return ranked_models[:top_n]


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_model_comparison(
    ranked_models: List[Dict[str, Any]],
    output_path: str = None
) -> None:
    """
    Visualize model comparison (rank vs score vs fit).
    
    Args:
        ranked_models: Output from rank_models()
        output_path: Save to file
    """
    if not ranked_models:
        print("No models to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Composite score by rank
    ax = axes[0, 0]
    ranks = [m["rank"] for m in ranked_models]
    scores = [m["composite_score"] for m in ranked_models]
    ax.barh(ranks, scores, color='steelblue')
    ax.set_xlabel('Composite Score')
    ax.set_ylabel('Model Rank')
    ax.set_title('Model Ranking by Composite Score')
    ax.invert_yaxis()
    
    # 2. Fit (R²) vs Ordinality scatter
    ax = axes[0, 1]
    fits = [m["fit_r2"] for m in ranked_models]
    ordinality = [m["ordinality"] for m in ranked_models]
    ax.scatter(fits, ordinality, s=200, alpha=0.6, c=scores, cmap='viridis')
    ax.set_xlabel('Model Fit (R²)')
    ax.set_ylabel('Ordinality Score')
    ax.set_title('Fit vs Ordinality')
    ax.grid(True, alpha=0.3)
    
    # 3. Top 5 components breakdown
    ax = axes[1, 0]
    top_5 = ranked_models[:5]
    model_names = [m["model_id"] for m in top_5]
    fit_scores = [m["fit_r2"] for m in top_5]
    x = np.arange(len(model_names))
    ax.bar(x, fit_scores, label='Fit (R²)', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Top 5 Models - Fit')
    ax.legend()
    
    # 4. Composite score distribution
    ax = axes[1, 1]
    ax.hist(scores, bins=10, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Composite Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution')
    ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean={np.mean(scores):.3f}')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison saved to {output_path}")
    
    return fig


# =============================================================================
# GENAI PREPARATION
# =============================================================================

def prepare_model_ranking_narrative(
    ranked_models: List[Dict[str, Any]]
) -> str:
    """
    Prepare ranked models for GenAI narrative generation.
    
    Args:
        ranked_models: Output from rank_models()
        
    Returns:
        Formatted text for LLM
    """
    prompt = "# TOP MODELS FOR NARRATIVE\n\n"
    
    for model in ranked_models[:10]:
        prompt += f"\n## Rank {model['rank']}: {model['model_id']}\n"
        prompt += f"- **Composite Score**: {model['composite_score']}\n"
        prompt += f"- **Model Fit (R²)**: {model['fit_r2']}\n"
        prompt += f"- **Ordinality Score**: {model['ordinality']}\n"
        
        if "coef" in model["model"]:
            prompt += f"- **Top Coefficients**:\n"
            coef_dict = model["model"]["coef"]
            sorted_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            for ch, coef in sorted_coef[:5]:
                prompt += f"  - {ch}: {coef:.4f}\n"
    
    return prompt
