"""
=============================================================================
EDA METRICS ENGINE
=============================================================================
Deterministic EDA computation: reach/frequency/engagement, correlations, overlap.

Pure functions, no visualization or side effects.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


# =============================================================================
# REACH/FREQUENCY/ENGAGEMENT
# =============================================================================

def calculate_reach_frequency_engagement(
    df: pd.DataFrame,
    channel_columns: List[str],
    sales_column: str = "total_gmv",
    segment_name: str = "National"
) -> Dict[str, Any]:
    """
    Calculate Reach, Frequency, Engagement metrics per channel.
    
    Reach = # periods channel was active (spend > 0)
    Frequency = Average spend per active period
    Engagement = Correlation with sales
    
    Args:
        df: Data with one row per period (month, day, etc.)
        channel_columns: List of channel spend columns
        sales_column: Name of sales/GMV column
        segment_name: Label for this segment
        
    Returns:
        {channel: {reach, reach_pct, frequency, engagement, ...}}
    """
    rfe_summary = {}
    
    if sales_column not in df.columns:
        # Find sales column by pattern
        sales_cols = [c for c in df.columns if 'gmv' in c.lower() or 'revenue' in c.lower() or 'sales' in c.lower()]
        if sales_cols:
            sales_column = sales_cols[0]
        else:
            raise ValueError(f"Sales column not found. Available: {df.columns.tolist()}")
    
    sales_series = df[sales_column].fillna(0)
    
    for channel in channel_columns:
        if channel not in df.columns:
            continue
        
        spend_series = df[channel].fillna(0)
        
        # Reach: # periods with non-zero spend
        reach = (spend_series > 0).sum()
        reach_pct = (reach / len(spend_series) * 100) if len(spend_series) > 0 else 0
        
        # Frequency: average spend per active period
        active_periods = spend_series[spend_series > 0]
        frequency = active_periods.mean() if len(active_periods) > 0 else 0
        
        # Engagement: correlation with sales
        engagement = spend_series.corr(sales_series)
        if pd.isna(engagement):
            engagement = 0
        
        rfe_summary[channel] = {
            "reach": int(reach),
            "reach_pct": round(reach_pct, 2),
            "frequency": round(frequency, 2),
            "engagement": round(engagement, 4),
            "total_spend": round(spend_series.sum(), 2),
            "avg_spend": round(spend_series.mean(), 2),
        }
    
    return rfe_summary


# =============================================================================
# SEGMENT EXTRACTION
# =============================================================================

def extract_segments(
    df: pd.DataFrame,
    segment_column: str = "Product_Category"
) -> Dict[str, pd.DataFrame]:
    """
    Extract data by segment.
    
    Args:
        df: DataFrame
        segment_column: Column to segment by
        
    Returns:
        {segment_name: segment_dataframe}
    """
    segments = {}
    
    if segment_column not in df.columns:
        return {"National": df}
    
    for segment in df[segment_column].unique():
        if pd.isna(segment):
            continue
        segments[str(segment)] = df[df[segment_column] == segment].copy()
    
    return segments


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def calculate_channel_sales_correlation(
    df: pd.DataFrame,
    channel_columns: List[str],
    sales_column: str = "total_gmv"
) -> Dict[str, float]:
    """
    Calculate correlation between each channel and sales.
    
    Args:
        df: Data
        channel_columns: Channel spend columns
        sales_column: Sales/GMV column
        
    Returns:
        {channel: correlation_coefficient}
    """
    correlations = {}
    
    if sales_column not in df.columns:
        return {}
    
    sales = df[sales_column].fillna(0)
    
    for channel in channel_columns:
        if channel not in df.columns:
            continue
        
        spend = df[channel].fillna(0)
        corr = spend.corr(sales)
        
        correlations[channel] = round(corr if not pd.isna(corr) else 0, 4)
    
    return correlations


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

def calculate_channel_overlap(
    df: pd.DataFrame,
    channel_columns: List[str],
    threshold: float = 0.0
) -> Dict[str, float]:
    """
    Calculate overlap between channels (% periods both active).
    
    Args:
        df: Data
        channel_columns: Channels to analyze
        threshold: Spend threshold for "active"
        
    Returns:
        {channel_pair: overlap_pct}
    """
    overlap = {}
    
    for i, ch1 in enumerate(channel_columns):
        if ch1 not in df.columns:
            continue
        
        active_ch1 = (df[ch1].fillna(0) > threshold).sum()
        
        for ch2 in channel_columns[i+1:]:
            if ch2 not in df.columns:
                continue
            
            active_ch2 = (df[ch2].fillna(0) > threshold).sum()
            both_active = ((df[ch1].fillna(0) > threshold) & (df[ch2].fillna(0) > threshold)).sum()
            
            total_periods = len(df)
            overlap_pct = (both_active / total_periods * 100) if total_periods > 0 else 0
            
            pair_name = f"{ch1}__{ch2}"
            overlap[pair_name] = round(overlap_pct, 2)
    
    return overlap


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_eda(
    df: pd.DataFrame,
    channel_columns: List[str],
    sales_column: str = "total_gmv",
    segment_column: str = None
) -> Dict[str, Any]:
    """
    Run complete EDA: reach/frequency/engagement, correlations, overlap.
    
    Args:
        df: Data
        channel_columns: Channel spend columns
        sales_column: Sales column
        segment_column: Optional column to segment by
        
    Returns:
        {
            'national': {rfe, correlations, overlap},
            'segments': {segment: {rfe, correlations, overlap}},
            'log': [messages]
        }
    """
    log = []
    results = {}
    
    try:
        # National level
        log.append("Computing national-level EDA...")
        results['national'] = {
            'rfe': calculate_reach_frequency_engagement(df, channel_columns, sales_column, 'National'),
            'correlations': calculate_channel_sales_correlation(df, channel_columns, sales_column),
            'overlap': calculate_channel_overlap(df, channel_columns),
        }
        log.append(f"  ✅ National: {len(results['national']['rfe'])} channels analyzed")
        
        # Segment level
        if segment_column and segment_column in df.columns:
            log.append(f"Extracting segments from {segment_column}...")
            segments = extract_segments(df, segment_column)
            results['segments'] = {}
            
            for seg_name, seg_df in segments.items():
                results['segments'][seg_name] = {
                    'rfe': calculate_reach_frequency_engagement(seg_df, channel_columns, sales_column, seg_name),
                    'correlations': calculate_channel_sales_correlation(seg_df, channel_columns, sales_column),
                    'overlap': calculate_channel_overlap(seg_df, channel_columns),
                }
                log.append(f"  ✅ Segment '{seg_name}': {len(results['segments'][seg_name]['rfe'])} channels")
        
        log.append("✅ EDA complete")
        
        return {
            'results': results,
            'log': log,
            'is_valid': True,
        }
    
    except Exception as e:
        log.append(f"❌ Error: {str(e)}")
        return {
            'results': results,
            'log': log,
            'is_valid': False,
            'error': str(e),
        }
