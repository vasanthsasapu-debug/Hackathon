"""
Pipeline Package - Core Modeling & Analysis

Modules:
  - column_classifier: Auto-classify CSV columns
  - eda: Exploratory data analysis (national + segment-level)
  - response_curves: Extract response curves & elasticity
  - models: Fit multiple model families & rank by composite score
"""

from .column_classifier import classify_columns, validate_classification
from .eda import (
    extract_segments,
    run_segment_eda,
    calculate_reach_frequency_engagement,
    analyze_channel_overlap,
    calculate_correlation_analysis,
)
from .response_curves import (
    run_response_curve_extraction,
    calculate_elasticity,
    visualize_response_curves,
)
from .models import (
    fit_glm_poisson,
    fit_glm_gamma,
    fit_fixed_effects,
    fit_mixed_effects,
    rank_models,
    visualize_model_comparison,
)

__all__ = [
    "classify_columns",
    "validate_classification",
    "extract_segments",
    "run_segment_eda",
    "calculate_reach_frequency_engagement",
    "analyze_channel_overlap",
    "calculate_correlation_analysis",
    "run_response_curve_extraction",
    "calculate_elasticity",
    "visualize_response_curves",
    "fit_glm_poisson",
    "fit_glm_gamma",
    "fit_fixed_effects",
    "fit_mixed_effects",
    "rank_models",
    "visualize_model_comparison",
]
