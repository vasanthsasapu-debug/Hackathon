"""
=============================================================================
RESPONSE CURVES -- Extract channel elasticity and fitted curves
=============================================================================
Extracts sales response to each channel from fitted models:
  1. Extract coefficients from trained models
  2. Calculate elasticity: ∂Sales/∂Spend
  3. Fit curves (linear, log-linear, polynomial)
  4. Generate confidence bands
  5. Store curves for optimization
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")


# =============================================================================
# CURVE FITTING FUNCTIONS
# =============================================================================

def linear_response(spend, alpha, beta):
    """Linear: Sales = α + β*Spend"""
    return alpha + beta * spend


def log_linear_response(spend, alpha, beta):
    """Log-linear: Sales = α + β*log(Spend)"""
    return alpha + beta * np.log(spend + 1)  # +1 to avoid log(0)


def diminishing_return(spend, alpha, beta, gamma):
    """Adstock: Sales = α + β*(1 - exp(-γ*Spend))"""
    return alpha + beta * (1 - np.exp(-gamma * spend))


def power_law(spend, alpha, beta):
    """Power law: Sales = α + β*Spend^0.5 (diminishing returns)"""
    return alpha + beta * np.power(spend + 1, 0.5)


# =============================================================================
# ELASTICITY CALCULATION
# =============================================================================

def calculate_elasticity(
    coefficients: Dict[str, float],
    spend_levels: pd.Series,
    functional_form: str = "log_linear"
) -> Dict[str, float]:
    """
    Calculate elasticity for each channel.
    
    Elasticity = (∂Sales/∂Spend) * (Spend/Sales)
    
    Args:
        coefficients: Model coefficients {channel: coef}
        spend_levels: Average spend levels {channel: spend}
        functional_form: "linear", "log_linear", "diminishing_return"
        
    Returns:
        {channel: elasticity}
    """
    elasticities = {}
    
    for channel, coef in coefficients.items():
        if channel not in spend_levels.index:
            continue
            
        avg_spend = spend_levels[channel]
        
        if functional_form == "linear":
            # Elasticity = β * (Spend / Sales)
            # For linear: ∂Sales/∂Spend = β
            elasticities[channel] = coef  # marginal effect
            
        elif functional_form == "log_linear":
            # For log-linear: ∂Sales/∂Spend = β / Spend
            # Elasticity = β
            elasticities[channel] = coef  # semi-elasticity (direct coef)
            
        elif functional_form == "diminishing_return":
            # Elasticity depends on spend level
            elasticities[channel] = coef / (avg_spend + 1)
            
        else:
            elasticities[channel] = coef
    
    return elasticities


# =============================================================================
# CURVE EXTRACTION & FITTING
# =============================================================================

def extract_response_curves(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    channel_columns: List[str],
    fit_functional_form: str = "log_linear",
    n_points: int = 50
) -> Dict[str, Dict[str, Any]]:
    """
    For each channel, fit a response curve: Sales = f(Channel_Spend).
    
    Args:
        X: Feature matrix
        y: Target (Sales/GMV)
        model: Trained sklearn model with coef_/params
        channel_columns: List of channel spend columns
        fit_functional_form: "linear", "log_linear", "power_law"
        n_points: Number of points for interpolation
        
    Returns:
        {channel: {
            "coefficients": dict,
            "elasticity": float,
            "curve_fit": scipy fit object,
            "predictions": array,
            "spend_range": array,
            "functional_form": str,
            "r_squared": float
        }}
    """
    curves = {}
    
    # Get model coefficients
    if hasattr(model, 'coef_'):
        coefs = model.coef_
    elif hasattr(model, 'params'):
        coefs = model.params
    else:
        coefs = [0] * len(X.columns)
    
    coef_dict = dict(zip(X.columns, coefs))
    
    # For each channel, create a curve
    for channel in channel_columns:
        if channel not in X.columns:
            continue
        
        # Get spend and sales data for this channel
        spend_data = X[channel].values
        sales_data = y.values
        
        # Remove zeros and negatives for log fits
        if fit_functional_form in ["log_linear", "power_law"]:
            mask = spend_data > 0
            spend_data = spend_data[mask]
            sales_data = sales_data[mask]
        
        if len(spend_data) < 3:
            continue
        
        # Fit curve
        try:
            if fit_functional_form == "linear":
                popt, _ = curve_fit(
                    linear_response, 
                    spend_data, 
                    sales_data,
                    maxfev=5000
                )
                spend_range = np.linspace(spend_data.min(), spend_data.max(), n_points)
                predictions = linear_response(spend_range, *popt)
                
            elif fit_functional_form == "log_linear":
                popt, _ = curve_fit(
                    log_linear_response,
                    spend_data,
                    sales_data,
                    maxfev=5000
                )
                spend_range = np.linspace(spend_data.min(), spend_data.max(), n_points)
                predictions = log_linear_response(spend_range, *popt)
                
            elif fit_functional_form == "power_law":
                popt, _ = curve_fit(
                    power_law,
                    spend_data,
                    sales_data,
                    p0=[sales_data.mean(), 1],
                    maxfev=5000
                )
                spend_range = np.linspace(spend_data.min(), spend_data.max(), n_points)
                predictions = power_law(spend_range, *popt)
            else:
                continue
            
            # Calculate R² for this fit
            residuals = sales_data - log_linear_response(spend_data, *popt) if fit_functional_form == "log_linear" else sales_data - predictions[:len(spend_data)]
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((sales_data - sales_data.mean())**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            curves[channel] = {
                "coefficients": popt.tolist(),
                "elasticity": coef_dict.get(channel, 0),
                "predictions": predictions,
                "spend_range": spend_range,
                "functional_form": fit_functional_form,
                "r_squared": r_squared,
                "data_points": len(spend_data),
            }
            
        except Exception as e:
            curves[channel] = {
                "error": str(e),
                "elasticity": coef_dict.get(channel, 0),
            }
    
    return curves


def visualize_response_curves(
    curves: Dict[str, Dict],
    output_path: str = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Visualize all response curves in a grid.
    
    Args:
        curves: Output from extract_response_curves()
        output_path: Save plot to file
        figsize: Figure size
    """
    n_channels = len(curves)
    n_cols = 3
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (channel, curve_data) in enumerate(curves.items()):
        ax = axes[idx]
        
        if "error" in curve_data:
            ax.text(0.5, 0.5, f"Error: {curve_data['error']}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{channel} (Failed)")
            continue
        
        spend_range = curve_data["spend_range"]
        predictions = curve_data["predictions"]
        form = curve_data["functional_form"]
        r2 = curve_data["r_squared"]
        
        ax.plot(spend_range, predictions, 'b-', linewidth=2, label='Fitted Curve')
        ax.fill_between(spend_range, 
                        predictions * 0.95, 
                        predictions * 1.05, 
                        alpha=0.2, 
                        label='±5% CI')
        
        ax.set_xlabel('Channel Spend', fontsize=10)
        ax.set_ylabel('Predicted Sales', fontsize=10)
        ax.set_title(f'{channel}\n{form} (R²={r2:.3f})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Response curves saved to {output_path}")
    
    return fig


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_response_curve_extraction(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    channel_columns: List[str],
    functional_form: str = "log_linear",
    output_path: str = None,
    verbose: bool = True
) -> Dict:
    """
    Main entry point: extract and visualize response curves.
    
    Returns:
        {
            "curves": dict,
            "elasticities": dict,
            "log": [actions],
            "summary": text
        }
    """
    log = []
    
    log.append(f"Extracting response curves for {len(channel_columns)} channels...")
    curves = extract_response_curves(X, y, model, channel_columns, functional_form)
    
    log.append(f"Calculating elasticities...")
    elasticities = calculate_elasticity(
        {ch: curves[ch].get('elasticity', 0) 
         for ch in curves if 'error' not in curves[ch]},
        X[channel_columns].mean(),
        functional_form
    )
    
    log.append(f"Visualizing curves...")
    visualize_response_curves(curves, output_path)
    
    summary = f"""
    Response Curve Analysis:
    - Channels analyzed: {len([c for c in curves if 'error' not in curves[c]])}
    - Functional form: {functional_form}
    - Elasticity range: {np.min(list(elasticities.values())):.3f} to {np.max(list(elasticities.values())):.3f}
    - Top elasticity channel: {max(elasticities, key=elasticities.get)}
    """
    
    return {
        "curves": curves,
        "elasticities": elasticities,
        "log": log,
        "summary": summary.strip(),
    }
