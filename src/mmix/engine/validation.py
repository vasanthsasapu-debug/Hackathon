"""
=============================================================================
VALIDATION ENGINE
=============================================================================
Fail-fast validation: schema, ordinality, metric consistency.

Validates outputs from each pipeline step to catch errors early.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    required_dtypes: Dict[str, str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame schema: columns exist, dtypes match.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must exist
        required_dtypes: Dict of {column: expected_dtype_name}
        
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")
    
    # Check dtypes
    if required_dtypes:
        for col, expected_dtype in required_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if expected_dtype not in actual_dtype:
                    errors.append(f"Column '{col}': expected {expected_dtype}, got {actual_dtype}")
    
    return len(errors) == 0, errors


# =============================================================================
# METRIC VALIDATION
# =============================================================================

def validate_metrics(
    data: Dict[str, Any],
    metric_bounds: Dict[str, Tuple[float, float]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate metric values are within expected bounds.
    
    Args:
        data: Dict of {metric_name: value}
        metric_bounds: Dict of {metric_name: (min, max)}
        
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    if metric_bounds is None:
        metric_bounds = {
            "r2": (0, 1),
            "correlation": (-1, 1),
            "elasticity": (0, 2),
        }
    
    for metric_name, bounds in metric_bounds.items():
        if metric_name in data:
            value = data[metric_name]
            min_val, max_val = bounds
            if not (min_val <= value <= max_val):
                errors.append(
                    f"Metric '{metric_name}': {value} outside bounds [{min_val}, {max_val}]"
                )
    
    return len(errors) == 0, errors


# =============================================================================
# ORDINALITY VALIDATION
# =============================================================================

def validate_ordinality(
    coefficients: Dict[str, float],
    channel_order: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that coefficients follow expected ordinality (order).
    
    Example:
        channel_order = ["TV", "Digital", "SEM", "Affiliates"]
        Expects: coef[TV] >= coef[Digital] >= coef[SEM] >= coef[Affiliates]
    
    Args:
        coefficients: Dict of {channel: coefficient}
        channel_order: List of channels in expected descending order
        
    Returns:
        (is_valid, list of violations)
    """
    violations = []
    
    for i in range(len(channel_order) - 1):
        ch1, ch2 = channel_order[i], channel_order[i + 1]
        
        if ch1 not in coefficients or ch2 not in coefficients:
            continue
        
        coef1, coef2 = coefficients[ch1], coefficients[ch2]
        
        if coef1 < coef2:
            violations.append(
                f"Ordinality violated: {ch1} ({coef1:.4f}) < {ch2} ({coef2:.4f})"
            )
    
    return len(violations) == 0, violations


# =============================================================================
# NUMERIC VALIDATION
# =============================================================================

def validate_numeric_column(
    series: pd.Series,
    allow_negative: bool = False,
    allow_zero: bool = True,
    allow_nan: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate numeric column values.
    
    Args:
        series: Pandas Series to validate
        allow_negative: Allow negative values
        allow_zero: Allow zero values
        allow_nan: Allow NaN values
        
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    if not pd.api.types.is_numeric_dtype(series):
        errors.append(f"Series is not numeric: {series.dtype}")
        return False, errors
    
    if series.isna().any() and not allow_nan:
        n_nan = series.isna().sum()
        errors.append(f"Series contains {n_nan} NaN values")
    
    if (series < 0).any() and not allow_negative:
        n_neg = (series < 0).sum()
        errors.append(f"Series contains {n_neg} negative values")
    
    if (series == 0).any() and not allow_zero:
        n_zero = (series == 0).sum()
        errors.append(f"Series contains {n_zero} zero values")
    
    return len(errors) == 0, errors


# =============================================================================
# CORRELATION VALIDATION
# =============================================================================

def validate_correlation_matrix(
    corr_matrix: pd.DataFrame,
    min_abs_value: float = -1.0,
    max_abs_value: float = 1.0
) -> Tuple[bool, List[str]]:
    """
    Validate correlation matrix structure and values.
    
    Args:
        corr_matrix: Correlation matrix (square, symmetric)
        min_abs_value: Minimum absolute value (default -1.0)
        max_abs_value: Maximum absolute value (default 1.0)
        
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    # Check shape
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        errors.append(f"Correlation matrix not square: {corr_matrix.shape}")
    
    # Check bounds
    if (corr_matrix.values < min_abs_value).any() or (corr_matrix.values > max_abs_value).any():
        errors.append("Correlation values outside [-1, 1]")
    
    # Check symmetry
    if not np.allclose(corr_matrix.values, corr_matrix.values.T):
        errors.append("Correlation matrix not symmetric")
    
    # Check diagonal is 1.0
    if not np.allclose(np.diag(corr_matrix.values), 1.0):
        errors.append("Correlation matrix diagonal is not all 1.0")
    
    return len(errors) == 0, errors


# =============================================================================
# OUTLIER DETECTION
# =============================================================================

def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> np.ndarray:
    """
    Detect outliers using Interquartile Range (IQR).
    
    Args:
        series: Pandas Series
        multiplier: IQR multiplier (default 1.5 for standard outliers)
        
    Returns:
        Boolean array (True = outlier)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers using Z-score.
    
    Args:
        series: Pandas Series
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        Boolean array (True = outlier)
    """
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


# =============================================================================
# COMPREHENSIVE VALIDATION
# =============================================================================

def validate_engine_output(
    output: Dict[str, Any],
    expected_keys: List[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate engine module output structure.
    
    Args:
        output: Output dict from engine module
        expected_keys: Required keys
        
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    if not isinstance(output, dict):
        errors.append(f"Output is not a dict: {type(output)}")
        return False, errors
    
    if expected_keys:
        missing = set(expected_keys) - set(output.keys())
        if missing:
            errors.append(f"Missing keys: {missing}")
    
    return len(errors) == 0, errors
