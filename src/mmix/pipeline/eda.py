"""
=============================================================================
ENHANCED EDA PIPELINE -- Segment-Level + National Analysis
=============================================================================
Adds to existing eda_pipeline.py:
  1. Segment-level extraction (Product_Category, Analytic_Category)
  2. Segment-level trend analysis
  3. Reach/Frequency/Engagement summaries per segment
  4. National + Segment correlation analysis
  5. GenAI-ready output (with prompts for narrative generation)
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (14, 6)
sns.set_style("whitegrid")


# =============================================================================
# SEGMENT EXTRACTION & DEFINITION
# =============================================================================

def extract_segments(
    monthly_df: pd.DataFrame,
    transaction_df: pd.DataFrame = None,
    segment_column: str = "Product_Category"
) -> Dict[str, pd.DataFrame]:
    """
    Extract data by segment (Product_Category, Analytic_Category, etc.).
    
    Args:
        monthly_df: Monthly aggregated data
        transaction_df: Optional transaction-level data
        segment_column: Which column to segment by
        
    Returns:
        {segment_name: segment_dataframe, ...}
    """
    segments = {}
    
    if segment_column in monthly_df.columns:
        for segment in monthly_df[segment_column].unique():
            if pd.isna(segment):
                continue
            segments[str(segment)] = monthly_df[monthly_df[segment_column] == segment].copy()
    
    return segments


# =============================================================================
# REACH/FREQUENCY/ENGAGEMENT ANALYSIS
# =============================================================================

def calculate_reach_frequency_engagement(
    monthly_df: pd.DataFrame,
    channel_columns: List[str],
    segment_name: str = "National"
) -> Dict[str, Any]:
    """
    Calculate Reach, Frequency, Engagement metrics per channel.
    
    Reach = # months channel was active (spend > 0)
    Frequency = Average spend per active month
    Engagement = Correlation with sales
    
    Args:
        monthly_df: Monthly data (one row per month, columns are channels)
        channel_columns: List of channel spend columns
        segment_name: Name of segment for labeling
        
    Returns:
        {channel: {reach, frequency, engagement, ...}}
    """
    rfe_summary = {}
    
    # Get sales column
    sales_col = [c for c in monthly_df.columns if 'gmv' in c.lower() or 'revenue' in c.lower()]
    if not sales_col:
        sales_col = [monthly_df.columns[-1]]  # Default to last column
    sales_col = sales_col[0]
    
    for channel in channel_columns:
        if channel not in monthly_df.columns:
            continue
        
        spend_series = monthly_df[channel].fillna(0)
        
        # Reach: # months with non-zero spend
        reach = (spend_series > 0).sum()
        reach_pct = (reach / len(spend_series)) * 100 if len(spend_series) > 0 else 0
        
        # Frequency: average spend per active month
        active_months = spend_series[spend_series > 0]
        frequency = active_months.mean() if len(active_months) > 0 else 0
        
        # Engagement: correlation with sales
        valid_idx = (spend_series > 0) & (monthly_df[sales_col] > 0)
        if valid_idx.sum() > 2:
            engagement = spend_series[valid_idx].corr(monthly_df[valid_idx][sales_col])
        else:
            engagement = np.nan
        
        rfe_summary[channel] = {
            "reach_months": int(reach),
            "reach_pct": round(reach_pct, 1),
            "frequency_avg": round(frequency, 2),
            "engagement_correlation": round(engagement, 3) if not np.isnan(engagement) else None,
            "total_spend": round(spend_series.sum(), 2),
            "min_spend": round(spend_series[spend_series > 0].min(), 2) if len(spend_series[spend_series > 0]) > 0 else 0,
            "max_spend": round(spend_series.max(), 2),
        }
    
    return rfe_summary


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

def analyze_channel_overlap(
    monthly_df: pd.DataFrame,
    channel_groups: Dict[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze overlap between channel groups (Personal vs Digital, etc.).
    
    Overlap = % months where both channel groups had activity
    
    Args:
        monthly_df: Monthly data
        channel_groups: {"traditional": [...], "digital": [...]}
        
    Returns:
        {
            "traditional_digital_overlap": float,
            "details": {...}
        }
    """
    if channel_groups is None:
        channel_groups = {
            "traditional": ["TV", "Sponsorship", "Radio"],
            "digital": ["Digital", "SEM", "Online.marketing", "Affiliates"],
        }
    
    overlap_analysis = {}
    
    for group1_name, group1_channels in channel_groups.items():
        for group2_name, group2_channels in channel_groups.items():
            if group1_name >= group2_name:  # Avoid duplicate pairs
                continue
            
            # Get active months for each group
            group1_active = (monthly_df[
                [c for c in group1_channels if c in monthly_df.columns]
            ] > 0).any(axis=1)
            
            group2_active = (monthly_df[
                [c for c in group2_channels if c in monthly_df.columns]
            ] > 0).any(axis=1)
            
            # Calculate overlap
            both_active = (group1_active & group2_active).sum()
            total_months = len(monthly_df)
            overlap_pct = (both_active / total_months) * 100 if total_months > 0 else 0
            
            overlap_key = f"{group1_name}_{group2_name}_overlap"
            overlap_analysis[overlap_key] = {
                "overlap_months": int(both_active),
                "overlap_pct": round(overlap_pct, 1),
                "total_months": int(total_months),
            }
    
    return overlap_analysis


# =============================================================================
# NATIONAL + SEGMENT CORRELATION
# =============================================================================

def calculate_correlation_analysis(
    monthly_df: pd.DataFrame,
    channel_columns: List[str],
    sales_column: str = "total_gmv",
    segment_name: str = "National"
) -> Dict[str, Any]:
    """
    Calculate channel-to-sales correlations at national and segment level.
    
    Args:
        monthly_df: Monthly data
        channel_columns: List of channel columns
        sales_column: Target column name
        segment_name: Name of segment
        
    Returns:
        {
            "correlations": {channel: corr_value, ...},
            "p_values": {channel: p_value, ...},
            "top_channels": [{channel, corr, ranking}, ...]
        }
    """
    from scipy.stats import pearsonr
    
    correlations = {}
    p_values = {}
    
    if sales_column not in monthly_df.columns:
        return {"error": f"{sales_column} not found"}
    
    sales_data = monthly_df[sales_column].fillna(0)
    
    for channel in channel_columns:
        if channel not in monthly_df.columns:
            continue
        
        channel_data = monthly_df[channel].fillna(0)
        
        # Remove any NaN pairs
        valid_idx = ~(sales_data.isna() | channel_data.isna())
        
        if valid_idx.sum() > 2:
            corr, p_val = pearsonr(sales_data[valid_idx], channel_data[valid_idx])
            correlations[channel] = round(corr, 3)
            p_values[channel] = round(p_val, 4)
        else:
            correlations[channel] = np.nan
            p_values[channel] = np.nan
    
    # Rank channels by correlation (descending)
    ranked = sorted(
        [(ch, corr) for ch, corr in correlations.items() if not np.isnan(corr)],
        key=lambda x: x[1],
        reverse=True
    )
    
    top_channels = [
        {"channel": ch, "correlation": corr, "rank": idx + 1}
        for idx, (ch, corr) in enumerate(ranked)
    ]
    
    return {
        "correlations": correlations,
        "p_values": p_values,
        "top_channels": top_channels,
        "segment": segment_name,
    }


# =============================================================================
# SEGMENT-LEVEL SUMMARY
# =============================================================================

def run_segment_eda(
    monthly_df: pd.DataFrame,
    channel_columns: List[str],
    segment_column: str = "Product_Category",
    sales_column: str = "total_gmv",
    channel_groups: Dict[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Run complete EDA at segment level.
    
    Returns:
        {
            "segments": {
                segment_name: {
                    "rfe": {...},
                    "correlations": {...},
                    "overlap": {...},
                    "summary": "text for GenAI"
                },
                ...
            },
            "national": {...},
            "log": [...]
        }
    """
    log = []
    segments_data = {}
    
    # National level
    log.append(f"Running national-level EDA...")
    national_rfe = calculate_reach_frequency_engagement(
        monthly_df, channel_columns, "National"
    )
    national_corr = calculate_correlation_analysis(
        monthly_df, channel_columns, sales_column, "National"
    )
    national_overlap = analyze_channel_overlap(monthly_df, channel_groups)
    
    segments_data["National"] = {
        "rfe": national_rfe,
        "correlations": national_corr,
        "overlap": national_overlap,
    }
    
    # Segment level
    if segment_column in monthly_df.columns:
        log.append(f"Running segment-level EDA by {segment_column}...")
        
        for segment in monthly_df[segment_column].unique():
            if pd.isna(segment):
                continue
            
            segment_name = str(segment)
            segment_df = monthly_df[monthly_df[segment_column] == segment]
            
            if len(segment_df) < 3:
                log.append(f"⚠️  Skipping {segment_name} (< 3 months data)")
                continue
            
            seg_rfe = calculate_reach_frequency_engagement(
                segment_df, channel_columns, segment_name
            )
            seg_corr = calculate_correlation_analysis(
                segment_df, channel_columns, sales_column, segment_name
            )
            seg_overlap = analyze_channel_overlap(segment_df, channel_groups)
            
            segments_data[segment_name] = {
                "rfe": seg_rfe,
                "correlations": seg_corr,
                "overlap": seg_overlap,
            }
    
    log.append(f"✅ EDA complete for {len(segments_data)} segments/levels")
    
    return {
        "segments": segments_data,
        "log": log,
    }


# =============================================================================
# GENAI NARRATIVE PREPARATION
# =============================================================================

def prepare_eda_narrative_input(eda_results: Dict[str, Any]) -> str:
    """
    Format EDA results into a prompt for GenAI narrative generation.
    
    Args:
        eda_results: Output from run_segment_eda()
        
    Returns:
        Formatted text for LLM
    """
    prompt = "# EDA FINDINGS TO NARRATE\n\n"
    
    for segment_name, seg_data in eda_results["segments"].items():
        prompt += f"\n## {segment_name} Level\n"
        
        # RFE Summary
        rfe = seg_data["rfe"]
        prompt += f"\n### Reach/Frequency/Engagement:\n"
        for channel, metrics in rfe.items():
            prompt += f"- **{channel}**: Reach={metrics['reach_pct']}%, Frequency=${metrics['frequency_avg']}, Correlation={metrics['engagement_correlation']}\n"
        
        # Top Correlations
        corr = seg_data["correlations"]
        if "top_channels" in corr:
            prompt += f"\n### Channel-Sales Correlations (Top 5):\n"
            for item in corr["top_channels"][:5]:
                prompt += f"- {item['channel']}: {item['correlation']} (rank {item['rank']})\n"
        
        # Overlap
        if seg_data["overlap"]:
            prompt += f"\n### Channel Overlap:\n"
            for overlap_type, metrics in seg_data["overlap"].items():
                prompt += f"- {overlap_type}: {metrics['overlap_pct']}%\n"
    
    return prompt
