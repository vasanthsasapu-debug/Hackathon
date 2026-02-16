"""
=============================================================================
UTILITIES ENGINE
=============================================================================
Centralized channel lookups, configuration helpers, aggregation utilities.

Single source of truth for channel lists and common operations.
"""

import pandas as pd
from typing import List, Dict, Any


# =============================================================================
# CHANNEL DEFINITIONS (Centralized)
# =============================================================================

# E-commerce marketing channels (from data context)
MARKETING_CHANNELS = {
    "TV": {"type": "Broadcast", "category": "Traditional Media"},
    "Digital": {"type": "Digital", "category": "Online"},
    "Sponsorship": {"type": "Sponsorship", "category": "Partnership"},
    "Content Marketing": {"type": "Content", "category": "Owned Media"},
    "Online marketing": {"type": "Digital", "category": "Online"},
    "Affiliates": {"type": "Performance", "category": "Partner Marketing"},
    "SEM": {"type": "Digital", "category": "Online"},
    "Radio": {"type": "Broadcast", "category": "Traditional Media"},
    "Email": {"type": "Email", "category": "Direct"},
    "SMS": {"type": "SMS", "category": "Direct"},
}

# Channel importance order (from problem domain - pharma template, adapted to e-comm)
CHANNEL_IMPORTANCE_ORDER = [
    "TV",
    "Digital",
    "SEM",
    "Sponsorship",
    "Content Marketing",
    "Online marketing",
    "Affiliates",
    "Radio",
    "Email",
    "SMS",
]

# Ordinality constraints (channels should follow this order in coefficient magnitude)
ORDINALITY_CHANNELS = [
    "TV",
    "Digital",
    "SEM",
    "Sponsorship",
    "Affiliates",
]


# =============================================================================
# CHANNEL UTILITIES
# =============================================================================

def get_marketing_channels() -> Dict[str, Dict[str, str]]:
    """Return centralized channel definitions."""
    return MARKETING_CHANNELS.copy()


def get_channel_importance_order() -> List[str]:
    """Return channels in importance order."""
    return CHANNEL_IMPORTANCE_ORDER.copy()


def get_ordinality_channels() -> List[str]:
    """Return channels that should follow ordinality constraints."""
    return ORDINALITY_CHANNELS.copy()


def get_channels_by_type(channel_type: str) -> List[str]:
    """
    Get channels of a specific type.
    
    Args:
        channel_type: "Broadcast", "Digital", "Direct", etc.
        
    Returns:
        List of channel names
    """
    return [
        ch for ch, meta in MARKETING_CHANNELS.items()
        if meta.get("type") == channel_type
    ]


def filter_channels_in_data(
    df: pd.DataFrame,
    classification: Dict[str, str]
) -> List[str]:
    """
    Find all promotional_activity columns in data.
    
    Args:
        df: DataFrame
        classification: Column classification dict
        
    Returns:
        List of column names classified as Promotional_Activity
    """
    return [
        col for col, cat in classification.items()
        if cat == "Promotional_Activity"
    ]


# =============================================================================
# AGGREGATION UTILITIES
# =============================================================================

def aggregate_by_segment(
    df: pd.DataFrame,
    segment_column: str,
    numeric_columns: List[str],
    agg_func: str = "sum"
) -> pd.DataFrame:
    """
    Aggregate numeric columns by segment.
    
    Args:
        df: DataFrame
        segment_column: Column to group by
        numeric_columns: Columns to aggregate
        agg_func: Aggregation function ("sum", "mean", "count")
        
    Returns:
        Aggregated DataFrame
    """
    if agg_func == "sum":
        return df.groupby(segment_column)[numeric_columns].sum()
    elif agg_func == "mean":
        return df.groupby(segment_column)[numeric_columns].mean()
    elif agg_func == "count":
        return df.groupby(segment_column)[numeric_columns].count()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")


def compute_percent_contribution(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Compute percent contribution for each column.
    
    Args:
        df: DataFrame
        columns: Columns to compute percentages for
        
    Returns:
        DataFrame with contribution percentages
    """
    result = df[columns].copy()
    total = result.sum(axis=1)
    
    for col in columns:
        result[col + "_pct"] = (result[col] / total * 100).round(2)
    
    return result


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def build_config_dict(
    channels: List[str] = None,
    ordinality_channels: List[str] = None,
    vif_threshold: float = 5.0,
    r2_threshold: float = 0.3,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Build unified config dict for pipeline.
    
    Args:
        channels: List of channels (uses default if None)
        ordinality_channels: Channels with ordinality constraints
        vif_threshold: Multicollinearity threshold
        r2_threshold: Minimum model fit
        cv_folds: Cross-validation folds
        
    Returns:
        Config dict
    """
    return {
        "channels": channels or list(MARKETING_CHANNELS.keys()),
        "ordinality_channels": ordinality_channels or ORDINALITY_CHANNELS,
        "vif_threshold": vif_threshold,
        "r2_threshold": r2_threshold,
        "cv_folds": cv_folds,
    }


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_channel_in_config(
    channel: str,
    channels: List[str] = None
) -> bool:
    """
    Check if channel is in known channels.
    
    Args:
        channel: Channel name
        channels: List of valid channels (uses default if None)
        
    Returns:
        True if valid
    """
    valid_channels = channels or list(MARKETING_CHANNELS.keys())
    return channel in valid_channels


def extract_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract numeric column names from DataFrame.
    
    Args:
        df: DataFrame
        
    Returns:
        List of numeric column names
    """
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()


def extract_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract categorical column names from DataFrame.
    
    Args:
        df: DataFrame
        
    Returns:
        List of categorical column names
    """
    return df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
