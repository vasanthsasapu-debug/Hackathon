"""
=============================================================================
CONFIG -- Centralized Configuration for MMIX Pipeline
=============================================================================
Manages:
  - Azure OpenAI credentials
  - Pipeline hyperparameters
  - Data column mappings
  - Model settings
  - Export preferences
=============================================================================
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Dict, List, Optional

load_dotenv()

# =============================================================================
# AZURE OPENAI CONFIGURATION
# =============================================================================

AZURE_ENDPOINT = os.getenv(
    "AZURE_ENDPOINT",
    "https://zs-eu1-ail-agentics-openai-team10.openai.azure.com/"
)
API_VERSION = os.getenv("API_VERSION", "2024-12-01-preview")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# =============================================================================
# PIPELINE HYPERPARAMETERS
# =============================================================================

# EDA Settings
SEGMENT_COLUMN = "Product_Category"  # Which column to segment by
MIN_SEGMENT_SIZE = 3  # Minimum rows per segment

# Feature Engineering
MAX_VIF = 5.0  # Variance Inflation Factor threshold
MIN_MODEL_R_SQUARED = 0.5  # Minimum R² to accept model
ORDINALITY_THRESHOLD = 0.8  # % of coefs that should have expected sign

# Modeling
RANDOM_STATE = 42
TEST_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5

# Optimization
MIN_BUDGET_ALLOCATION = 0.02  # Min 2% per channel
MAX_BUDGET_ALLOCATION = 0.5   # Max 50% per channel
ELASTICITY_BOUNDS = (-2.0, 2.0)  # Elasticity range

# Response Curves
RESPONSE_CURVE_METHODS = ["linear", "log_linear", "power_law", "diminishing_return"]
DEFAULT_RESPONSE_CURVE = "power_law"

# =============================================================================
# DATA COLUMN MAPPINGS
# =============================================================================

# Expected column categories (auto-populated by column_classifier)
REQUIRED_COLUMN_CATEGORIES = [
    "Time_Stamp",
    "Sales_Output",
    "Promotional_Activity",
]

OPTIONAL_COLUMN_CATEGORIES = [
    "Entity_ID",
    "Brand_Health",
    "Demographic_Segment",
    "Discount_Feature",
]

# Default channel names (if not auto-detected)
DEFAULT_CHANNELS = [
    "TV", "Digital", "SEM", "Email", "Sponsorship", "Affiliates", "Radio", "Other"
]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Supported model families
MODEL_FAMILIES = {
    "ridge": {"estimator": "Ridge", "hyperparams": {"alpha": 1.0}},
    "bayesian_ridge": {"estimator": "BayesianRidge", "hyperparams": {}},
    "glm_poisson": {"estimator": "GLMPoisson", "hyperparams": {}},
    "glm_gamma": {"estimator": "GLMGamma", "hyperparams": {}},
    "fixed_effects": {"estimator": "FixedEffects", "hyperparams": {}},
    "mixed_effects": {"estimator": "MixedEffects", "hyperparams": {}},
}

# Composite scoring weights
MODEL_SCORING_WEIGHTS = {
    "fit": 0.5,          # R² / RMSE
    "stability": 0.3,    # Residual variance
    "ordinality": 0.2,   # Expected signs
}

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

# Excel export settings
EXCEL_SHEET_NAMES = {
    "eda": "EDA Summary",
    "models": "Model Rankings",
    "curves": "Response Curves",
    "optimization": "Optimization Scenarios",
    "narratives": "GenAI Insights",
}

# PowerPoint settings
PPT_THEME_COLOR = (25, 55, 109)  # Dark blue RGB
PPT_ACCENT_COLOR = (242, 169, 0)  # Gold RGB
PPT_NUM_SLIDES = 10

# =============================================================================
# LLM SETTINGS
# =============================================================================

LLM_SETTINGS = {
    "temperature": 0.3,      # Lower = more focused
    "max_tokens": 1000,
    "top_p": 0.95,
}

# System prompts for different narrative types
SYSTEM_PROMPTS = {
    "default": (
        "You are a senior marketing analytics consultant specializing in "
        "Marketing Mix Modeling (MMIX) for e-commerce. You explain data findings "
        "in clear, actionable business language. Be concise but insightful. "
        "Always highlight: what happened, why it matters, and what to do about it."
    ),
    "eda": (
        "You are analyzing exploratory data for a marketing campaign. "
        "Focus on: segment differences, seasonal trends, channel strengths, "
        "and surprising patterns. Be specific with numbers."
    ),
    "modeling": (
        "You are explaining statistical model results. "
        "Highlight: model quality, most important channels, coefficient signs, "
        "and statistical significance. Avoid jargon."
    ),
    "optimization": (
        "You are comparing budget allocation scenarios. "
        "For each scenario, explain: changes from baseline, expected uplift, "
        "risk/reward trade-offs, and top 3 recommendations."
    ),
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_llm_client() -> AzureOpenAI:
    """
    Initialize and return Azure OpenAI client.
    
    Raises:
        ValueError: If AZURE_OPENAI_API_KEY not in environment
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ AZURE_OPENAI_API_KEY not found in environment.\n"
            "   Create a .env file in the project root with:\n"
            "   AZURE_OPENAI_API_KEY=your_key_here"
        )

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=api_key,
        api_version=API_VERSION
    )
    return client


def get_system_prompt(prompt_type: str = "default") -> str:
    """Get system prompt by type."""
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["default"])


def validate_configuration() -> Dict[str, bool]:
    """
    Validate that all required configuration is available.
    
    Returns:
        {config_name: is_valid}
    """
    checks = {
        "azure_endpoint": bool(AZURE_ENDPOINT),
        "api_version": bool(API_VERSION),
        "deployment_name": bool(DEPLOYMENT_NAME),
        "azure_api_key": bool(os.getenv("AZURE_OPENAI_API_KEY")),
        "hyperparameters": bool(RANDOM_STATE is not None),
        "model_families": len(MODEL_FAMILIES) > 0,
        "channels": len(DEFAULT_CHANNELS) > 0,
    }
    return checks


def test_llm_connection() -> bool:
    """
    Quick test to verify Azure OpenAI connection works.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Connection successful!' in exactly one sentence.",
                }
            ],
            max_tokens=50,
            temperature=0.1,
        )
        print(f"✅ Azure OpenAI connection verified")
        return True
    except Exception as e:
        print(f"❌ LLM connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("MMIX Configuration Check")
    print("=" * 70)
    
    checks = validate_configuration()
    for key, value in checks.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")
    
    print("\nTesting LLM connection...")
    test_llm_connection()
