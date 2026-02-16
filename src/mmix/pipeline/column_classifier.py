"""
=============================================================================
COLUMN CLASSIFIER -- Auto-assign column categories
=============================================================================
Uses pattern matching + LLM to automatically classify columns into:
  - Time_Stamp
  - Entity_ID
  - Sales_Output
  - Promotional_Activity (channel spend)
  - Brand_Health
  - Demographic_Segment
=============================================================================
"""

import pandas as pd
import re
from typing import Dict, List, Tuple


# =============================================================================
# PATTERN-BASED CLASSIFIER (Heuristic)
# =============================================================================

COLUMN_PATTERNS = {
    "Time_Stamp": [
        r"date|month|year|time|timestamp|period",
        r"^\d{4}-\d{2}|^\d{2}-\d{2}-\d{4}",  # Date-like strings
    ],
    "Entity_ID": [
        r"id|identifier|code|key",
        r"^(HCP|customer|product|segment)_id",
    ],
    "Sales_Output": [
        r"gmv|sales|revenue|gross|total_gmv|sales_value|net_sales",
        r"turnover|income|proceeds",
    ],
    "Promotional_Activity": [
        r"tv|digital|sponsorship|sem|email|calls|visits|rep|affiliates|"
        r"content|online|marketing|spend|investment|budget|allocation",
    ],
    "Brand_Health": [
        r"nps|satisfaction|awareness|brand|sentiment|health|score|rating",
    ],
    "Demographic_Segment": [
        r"category|segment|group|class|tier|region|geography|product_category|analytic|sub_",
    ],
    "Discount_Feature": [
        r"discount|discount_pct|discount_amount|sale|promotion",
    ],
}


def classify_columns(df: pd.DataFrame, verbose: bool = False) -> Dict[str, str]:
    """
    Classify DataFrame columns into semantic categories.
    
    Args:
        df: Input DataFrame
        verbose: Print classification details
        
    Returns:
        Dict mapping column name → category
    """
    classification = {}
    
    for col in df.columns:
        col_lower = col.lower()
        matched_category = None
        
        # Try to match against patterns
        for category, patterns in COLUMN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    matched_category = category
                    break
            if matched_category:
                break
        
        # Default to "Numeric" or "Categorical" if no match
        if matched_category is None:
            if df[col].dtype in ['float64', 'int64']:
                matched_category = "Numeric_Feature"
            else:
                matched_category = "Categorical_Feature"
        
        classification[col] = matched_category
        
        if verbose:
            print(f"  {col:30s} → {matched_category}")
    
    return classification


def generate_column_summary(classification: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Group columns by category.
    
    Args:
        classification: Output from classify_columns()
        
    Returns:
        Dict mapping category → [list of columns]
    """
    summary = {}
    for col, cat in classification.items():
        if cat not in summary:
            summary[cat] = []
        summary[cat].append(col)
    
    return summary


def validate_classification(
    classification: Dict[str, str], 
    required_categories: List[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that classification has required categories.
    
    Args:
        classification: Output from classify_columns()
        required_categories: Expected categories (if None, defaults to essential)
        
    Returns:
        (is_valid, list of missing categories)
    """
    if required_categories is None:
        required_categories = [
            "Time_Stamp",
            "Sales_Output",
            "Promotional_Activity",
        ]
    
    found_categories = set(classification.values())
    missing = set(required_categories) - found_categories
    
    return len(missing) == 0, list(missing)


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_column_classification(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Main entry point: classify columns and return structured output.
    
    Returns:
        {
            "classification": dict,
            "summary": dict,
            "is_valid": bool,
            "missing_categories": list,
            "log": [actions taken]
        }
    """
    log = []
    
    log.append(f"Classifying {len(df.columns)} columns...")
    classification = classify_columns(df, verbose=verbose)
    
    log.append(f"Grouping columns by category...")
    summary = generate_column_summary(classification)
    
    log.append(f"Validating classification...")
    is_valid, missing = validate_classification(classification)
    
    if not is_valid:
        log.append(f"⚠️  Missing categories: {missing}")
    else:
        log.append("✅ All required categories found")
    
    return {
        "classification": classification,
        "summary": summary,
        "is_valid": is_valid,
        "missing_categories": missing,
        "log": log,
    }
