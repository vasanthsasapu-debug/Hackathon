"""
=============================================================================
MMIX ENGINE -- Deterministic, testable, reproducible computation layer
=============================================================================

All modules are deterministic and stateless. No side effects except explicit
file writes to outputs/.

Modules:
  - column_classification: Auto-classify columns into semantic categories
  - eda_metrics: Reach/frequency/engagement, correlations, overlap analysis
  - modeling: Incremental feature selection, GLM, Bayesian, Mixed Effects
  - response_curves: Fit curves, compute elasticity, confidence bands
  - optimization_engine: Budget allocation scenarios (Base, Neutral, Max, Sky)
  - goal_parsing: NLP-based constraint extraction from user prompts
  - validation: Schema, ordinality, metric consistency checks
  - utilities: Channel lookups, aggregation helpers

Separation of Concerns:
  - All computation deterministic (no randomness unless seeded explicitly)
  - All functions pure (input → output, no globals)
  - All outputs validated before return
  - All errors fail-fast with clear messages

TODO (V2):
  - Move existing pipeline/ modules (eda_enhanced, advanced_modeling, etc.) here
  - Create eda_metrics, modeling, response_curves, optimization_engine, goal_parsing
  - Refactor existing code to use engine modules
"""

from . import column_classification
from . import eda_metrics
from . import modeling
from . import response_curves
from . import optimization_engine
from . import goal_parsing
from . import validation
from . import utilities

__all__ = [
    "column_classification",
    "eda_metrics",
    "modeling",
    "response_curves",
    "optimization_engine",
    "goal_parsing",
    "validation",
    "utilities",
]
