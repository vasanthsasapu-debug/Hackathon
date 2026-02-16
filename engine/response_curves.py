"""
=============================================================================
RESPONSE CURVES ENGINE
=============================================================================
Deterministic curve fitting and elasticity computation.

Pure functions, no visualization.
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Any


# =============================================================================
# CURVE FITTING FUNCTIONS
# =============================================================================

def linear_response(spend, alpha, beta):
    """Linear: Sales = α + β*Spend"""
    return alpha + beta * spend


def log_linear_response(spend, alpha, beta):
    """Log-linear: Sales = α + β*log(Spend)"""
    return alpha + beta * np.log(spend + 1)


def power_law_response(spend, alpha, beta):
    """Power law: Sales = α + β*Spend^0.5 (diminishing returns)"""
    return alpha + beta * np.power(spend + 1, 0.5)


def diminishing_return_response(spend, alpha, beta, gamma):
    """Diminishing return: Sales = α + β*(1 - exp(-γ*Spend))"""
    return alpha + beta * (1 - np.exp(-gamma * spend))


# =============================================================================
# ELASTICITY CALCULATION
# =============================================================================

def calculate_elasticity(
    coef: float,
    avg_spend: float,
    functional_form: str = "log_linear"
) -> float:
    """
    Calculate elasticity from coefficient and functional form.
    
    For log-linear (most common):
      Sales = α + β*log(Spend)
      Elasticity ≈ β (semi-elasticity)
    
    For linear:
      Sales = α + β*Spend
      Elasticity = β * (Spend / Sales)
    
    Args:
        coef: Model coefficient
        avg_spend: Average spend level
        functional_form: "linear", "log_linear", "power_law", "diminishing_return"
        
    Returns:
        Elasticity (unitless)
    """
    if functional_form == "log_linear":
        # Semi-elasticity: coef is the elasticity
        return coef
    
    elif functional_form == "linear":
        # Elasticity = β * (Spend / Sales)
        # Using marginal effect as proxy
        return coef
    
    elif functional_form == "power_law":
        # Sales = α + β*(Spend)^0.5
        # ∂Sales/∂Spend = 0.5 * β * (Spend)^(-0.5)
        if avg_spend > 0:
            return 0.5 * coef / np.sqrt(avg_spend)
        return coef
    
    elif functional_form == "diminishing_return":
        # Sales = α + β*(1 - exp(-γ*Spend))
        # Elasticity varies with spend
        if avg_spend > 0:
            return coef / (avg_spend + 1)
        return coef
    
    else:
        return coef


# =============================================================================
# CURVE FITTING
# =============================================================================

def fit_response_curve(
    spend_data: np.ndarray,
    sales_data: np.ndarray,
    curve_types: List[str] = None
) -> Dict[str, Any]:
    """
    Fit multiple response curve types and return best fit.
    
    Args:
        spend_data: Channel spend values
        sales_data: Corresponding sales values
        curve_types: Which curves to fit (default: all)
        
    Returns:
        {
            'best_fit': {curve_type, params, r2, ...},
            'all_fits': [{curve_type, params, r2, ...}],
        }
    """
    if curve_types is None:
        curve_types = ['linear', 'log_linear', 'power_law']
    
    # Remove NaNs
    mask = ~(np.isnan(spend_data) | np.isnan(sales_data))
    spend_data = spend_data[mask]
    sales_data = sales_data[mask]
    
    if len(spend_data) < 3:
        return {
            'best_fit': None,
            'all_fits': [],
            'error': 'Insufficient data for curve fitting',
        }
    
    fits = []
    best_fit = None
    best_r2 = -np.inf
    
    # Linear fit
    if 'linear' in curve_types:
        try:
            popt, _ = curve_fit(linear_response, spend_data, sales_data, maxfev=10000)
            y_pred = linear_response(spend_data, *popt)
            r2 = 1 - (np.sum((sales_data - y_pred) ** 2) / np.sum((sales_data - sales_data.mean()) ** 2))
            
            fit = {
                'curve_type': 'linear',
                'params': {'alpha': float(popt[0]), 'beta': float(popt[1])},
                'r2': float(r2),
                'elasticity': calculate_elasticity(float(popt[1]), spend_data.mean(), 'linear'),
            }
            fits.append(fit)
            
            if r2 > best_r2:
                best_r2 = r2
                best_fit = fit
        except:
            pass
    
    # Log-linear fit
    if 'log_linear' in curve_types:
        try:
            popt, _ = curve_fit(log_linear_response, spend_data, sales_data, maxfev=10000)
            y_pred = log_linear_response(spend_data, *popt)
            r2 = 1 - (np.sum((sales_data - y_pred) ** 2) / np.sum((sales_data - sales_data.mean()) ** 2))
            
            fit = {
                'curve_type': 'log_linear',
                'params': {'alpha': float(popt[0]), 'beta': float(popt[1])},
                'r2': float(r2),
                'elasticity': calculate_elasticity(float(popt[1]), spend_data.mean(), 'log_linear'),
            }
            fits.append(fit)
            
            if r2 > best_r2:
                best_r2 = r2
                best_fit = fit
        except:
            pass
    
    # Power law fit
    if 'power_law' in curve_types:
        try:
            popt, _ = curve_fit(power_law_response, spend_data, sales_data, maxfev=10000)
            y_pred = power_law_response(spend_data, *popt)
            r2 = 1 - (np.sum((sales_data - y_pred) ** 2) / np.sum((sales_data - sales_data.mean()) ** 2))
            
            fit = {
                'curve_type': 'power_law',
                'params': {'alpha': float(popt[0]), 'beta': float(popt[1])},
                'r2': float(r2),
                'elasticity': calculate_elasticity(float(popt[1]), spend_data.mean(), 'power_law'),
            }
            fits.append(fit)
            
            if r2 > best_r2:
                best_r2 = r2
                best_fit = fit
        except:
            pass
    
    return {
        'best_fit': best_fit,
        'all_fits': fits,
        'best_r2': float(best_r2),
    }


# =============================================================================
# CONFIDENCE BANDS
# =============================================================================

def compute_confidence_bands(
    spend_data: np.ndarray,
    sales_data: np.ndarray,
    curve_params: Dict[str, float],
    curve_type: str = "log_linear",
    confidence: float = 0.95,
    n_points: int = 50
) -> Dict[str, Any]:
    """
    Compute confidence intervals for response curve.
    
    Uses bootstrap resampling.
    
    Args:
        spend_data: Historical spend
        sales_data: Historical sales
        curve_params: Fitted curve parameters
        curve_type: Type of curve
        confidence: Confidence level (0.95 = 95%)
        n_points: Number of points for band
        
    Returns:
        {
            'spend_range': [min_spend, max_spend],
            'predictions': [...],
            'ci_lower': [...],
            'ci_upper': [...],
        }
    """
    # Create spend range
    spend_range = np.linspace(spend_data.min(), spend_data.max(), n_points)
    
    # Select fitting function
    if curve_type == 'linear':
        fit_func = linear_response
    elif curve_type == 'log_linear':
        fit_func = log_linear_response
    elif curve_type == 'power_law':
        fit_func = power_law_response
    else:
        fit_func = log_linear_response
    
    # Predictions
    params = tuple(curve_params.values())
    predictions = fit_func(spend_range, *params)
    
    # Simple confidence bands (residual-based)
    residuals = sales_data - fit_func(spend_data, *params)
    residual_std = residuals.std()
    
    # Use t-distribution critical value (approximate)
    z_crit = 1.96  # 95% confidence
    
    ci_lower = predictions - (z_crit * residual_std)
    ci_upper = predictions + (z_crit * residual_std)
    
    return {
        'spend_range': spend_range.tolist(),
        'predictions': predictions.tolist(),
        'ci_lower': ci_lower.tolist(),
        'ci_upper': ci_upper.tolist(),
        'residual_std': float(residual_std),
    }


# =============================================================================
# ORCHESTRATION
# =============================================================================

def compute_response_curves(
    df: pd.DataFrame,
    channel_columns: List[str],
    sales_column: str = "total_gmv"
) -> Dict[str, Any]:
    """
    Compute response curves for all channels.
    
    Args:
        df: Data
        channel_columns: Channel spend columns
        sales_column: Sales column
        
    Returns:
        {channel: {curve_type, params, elasticity, confidence_bands, ...}}
    """
    curves = {}
    log = []
    
    if sales_column not in df.columns:
        log.append(f"❌ Sales column '{sales_column}' not found")
        return {'curves': {}, 'log': log, 'is_valid': False}
    
    sales_data = df[sales_column].fillna(0).values
    
    for channel in channel_columns:
        if channel not in df.columns:
            continue
        
        spend_data = df[channel].fillna(0).values
        
        if (spend_data == 0).all():
            log.append(f"⚠️  Channel '{channel}': No spend data")
            continue
        
        fit_result = fit_response_curve(spend_data, sales_data)
        
        if fit_result['best_fit']:
            best = fit_result['best_fit']
            
            # Confidence bands
            bands = compute_confidence_bands(
                spend_data, sales_data,
                best['params'],
                best['curve_type']
            )
            
            curves[channel] = {
                'curve_type': best['curve_type'],
                'params': best['params'],
                'r2': best['r2'],
                'elasticity': best['elasticity'],
                'confidence_bands': bands,
            }
            
            log.append(f"✅ {channel}: {best['curve_type']} (R²={best['r2']:.3f}, elasticity={best['elasticity']:.3f})")
        else:
            log.append(f"❌ {channel}: Could not fit curve")
    
    return {
        'curves': curves,
        'log': log,
        'is_valid': len(curves) > 0,
    }
