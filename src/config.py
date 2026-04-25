"""
=============================================================================
config.py -- Pipeline Configuration
=============================================================================
Single source of truth for paths, constants, file mappings, model settings,
LLM config, and the pipeline summary collector.

All other modules import from here. No hardcoded paths or constants elsewhere.
=============================================================================
"""

import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional -- keys can be set in environment directly

# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(name="mmix", level=logging.INFO):
    """Create a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = setup_logger()

# =============================================================================
# PATHS
# =============================================================================

def get_project_root():
    """Return project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_paths(data_dir=None, output_dir=None):
    """
    Resolve all directory paths. Accepts overrides or auto-detects.

    Args:
        data_dir:   override for data folder
        output_dir: override for output folder

    Returns:
        dict with keys: root, data_dir, output_dir, plots_dir, reports_dir
    """
    root = get_project_root()
    _data = data_dir or os.path.join(root, "data")
    _output = output_dir or os.path.join(root, "outputs")
    return {
        "root": root,
        "data_dir": _data,
        "output_dir": _output,
        "plots_dir": os.path.join(_output, "plots"),
        "reports_dir": os.path.join(_output, "reports"),
    }

# =============================================================================
# DATA FILE REGISTRY
# =============================================================================

DATA_FILES = {
    "transactions":  {"filename": "firstfile.csv",       "date_cols": ["Date"], "sep": ","},
    "sales":         {"filename": "Sales.csv",            "date_cols": ["Date"], "sep": "\t",
                      "date_format": "%d-%m-%Y %H:%M"},
    "monthly":       {"filename": "SecondFile.csv",       "date_cols": ["Date"], "sep": ","},
    "special_sales": {"filename": "SpecialSale.csv",      "date_cols": ["Date"], "sep": ","},
    "investment":    {"filename": "MediaInvestment.csv",   "date_cols": None,     "sep": ","},
    "nps":           {"filename": "MonthlyNPSscore.csv",  "date_cols": ["Date"], "sep": ","},
    "products":      {"filename": "ProductList.csv",      "date_cols": None,     "sep": ","},
}

# =============================================================================
# MEDIA CHANNELS & GROUPS
# =============================================================================

MEDIA_CHANNELS = [
    "TV", "Digital", "Sponsorship", "Content.Marketing",
    "Online.marketing", "Affiliates", "SEM", "Radio", "Other",
]

CHANNEL_GROUPS = {
    "traditional":          ["TV", "Sponsorship"],
    "digital_performance":  ["Online.marketing", "Affiliates"],
    "digital_brand":        ["Digital", "Content.Marketing", "SEM"],
    "other":                ["Radio", "Other"],
}
# Grouping rationale (from EDA correlation analysis):
#   - Online.marketing ↔ Affiliates: r=0.99 → must group
#   - Digital ↔ Content.Marketing ↔ SEM: r=0.90-0.97 → group together
#   - SEM moved from digital_performance to digital_brand based on
#     correlation evidence (SEM↔Digital r=0.97, SEM↔Content.Mkt r=0.96)
#   - Radio, Other: both strongly negative with GMV (r=-0.95, -0.96)
#   - TV, Sponsorship: positive, moderate cross-correlation (r=0.55)

# Log feature name -> raw column name mapping
LOG_TO_RAW_MAP = {
    "log_TV": "TV",
    "log_Digital": "Digital",
    "log_Sponsorship": "Sponsorship",
    "log_Content.Marketing": "Content.Marketing",
    "log_Online.marketing": "Online.marketing",
    "log_Affiliates": "Affiliates",
    "log_SEM": "SEM",
    "log_Radio": "Radio",
    "log_Other": "Other",
    "log_Total_Investment": "Total.Investment",
    "log_total_Discount": "total_Discount",
    "log_spend_traditional": "spend_traditional",
    "log_spend_digital_performance": "spend_digital_performance",
    "log_spend_digital_brand": "spend_digital_brand",
    "log_spend_other": "spend_other",
}

# =============================================================================
# MODEL SETTINGS
# =============================================================================

MODEL_SETTINGS = {
    "max_predictors_monthly": 3,
    "max_predictors_weekly": 6,
    "vif_threshold": 10,
    "ordinality_required": True,
    "zscore_threshold": 2.5,
    "iqr_multiplier": 1.5,
    "score_weights": {
        "fit": 0.35,
        "stability": 0.25,
        "ordinality": 0.25,
        "vif": 0.15,
    },
    # Quality gate thresholds used by QualityEvaluator (agent loop)
    "quality_thresholds": {
        "min_r2": 0.50,            # Weekly R² floor — below triggers RETRY
        "min_adj_r2": 0.45,        # Adjusted R² floor
        "max_vif": 50,             # Maximum acceptable VIF for best model
        "min_models_passed": 5,    # Minimum successful model specs
        "min_ordinality_rate": 0.5, # Fraction of convergence-confirmed channels
    },
}

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

LLM_CONFIG = {
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
    "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    "embedding_deployment": os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
}


def get_llm_api_key():
    """Return the LLM API key from the environment, or None."""
    key = LLM_CONFIG["api_key"]
    if not key:
        logger.warning("AZURE_OPENAI_API_KEY not set in environment")
    return key

# =============================================================================
# PIPELINE SUMMARY COLLECTOR
# =============================================================================

class PipelineSummary:
    """
    Collects structured summaries from every pipeline step.
    GenAI narrative module reads this to produce business reports.
    """

    def __init__(self):
        self.steps = {}
        self.errors = []
        self.warnings = []

    def add_step(self, step_name, summary, log=None, decisions=None):
        """Record a pipeline step's output."""
        self.steps[step_name] = {
            "summary": summary,
            "log": log or [],
            "decisions": decisions or [],
        }

    def add_error(self, step_name, error_msg):
        """Record a pipeline error."""
        self.errors.append({"step": step_name, "error": str(error_msg)})

    def add_warning(self, step_name, warning_msg):
        """Record a pipeline warning."""
        self.warnings.append({"step": step_name, "warning": str(warning_msg)})

    def get_full_summary(self):
        """
        Return the complete summary as a single string.
        This is the primary input for LLM narrative generation.
        """
        parts = []
        for name, data in self.steps.items():
            parts.append(f"## {name}\n{data['summary']}")
        if self.warnings:
            parts.append("\n## Warnings")
            for w in self.warnings:
                parts.append(f"- [{w['step']}] {w['warning']}")
        if self.errors:
            parts.append("\n## Errors")
            for e in self.errors:
                parts.append(f"- [{e['step']}] {e['error']}")
        return "\n\n".join(parts)

    def get_step_summary(self, step_name):
        """Return the summary string for one step, or empty string."""
        return self.steps.get(step_name, {}).get("summary", "")

    def __repr__(self):
        return (f"PipelineSummary(steps={len(self.steps)}, "
                f"errors={len(self.errors)}, warnings={len(self.warnings)})")


# =============================================================================
# SHARED HELPERS (used across multiple modules)
# =============================================================================

def find_col(df, candidates):
    """Return the first column name from *candidates* that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_channel_cols(df):
    """Return the subset of MEDIA_CHANNELS that exist in *df*."""
    return [c for c in MEDIA_CHANNELS if c in df.columns]