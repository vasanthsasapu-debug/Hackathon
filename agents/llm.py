"""
=============================================================================
ENHANCED LLM INTEGRATION -- GenAI narrative generation for all pipeline steps
=============================================================================
Expands llm_config.py with step-specific narrative generators:
  - EDA narratives
  - Outlier removal rationale
  - Feature engineering explanations
  - Model performance summaries
  - Model ranking recommendations
  - Optimization scenario comparisons
=============================================================================
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Dict, Any, Optional, List
import json

load_dotenv()


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

def get_llm_client() -> AzureOpenAI:
    """Initialize and return Azure OpenAI client."""
    api_key = os.getenv("AZURE_OPENAI_KEY")
    if not api_key:
        raise ValueError(
            "❌ AZURE_OPENAI_KEY not found in environment.\n"
            "   Create a .env file in the project root with:\n"
            "   AZURE_OPENAI_KEY=your_key_here"
        )
    
    endpoint = os.getenv("AZURE_ENDPOINT", "https://zs-eu1-ail-agentics-openai-team10.openai.azure.com/")
    api_version = os.getenv("API_VERSION", "2024-12-01-preview")
    
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )
    
    print("✅ Azure OpenAI client initialized.")
    return client


# =============================================================================
# STEP-SPECIFIC NARRATIVE GENERATORS
# =============================================================================

def generate_eda_narrative(
    client: AzureOpenAI,
    eda_prompt: str,
    segment_name: str = "National",
    max_tokens: int = 800
) -> str:
    """
    Generate narrative for EDA findings.
    
    Args:
        client: AzureOpenAI client
        eda_prompt: Formatted EDA data (from eda_enhanced.prepare_eda_narrative_input)
        segment_name: "National" or segment name
        max_tokens: Response length
        
    Returns:
        str: Narrative text
    """
    system_prompt = (
        "You are a senior marketing analytics consultant. Your task is to explain "
        "marketing channel performance data in clear, actionable insights. "
        "Focus on: (1) Which channels drive revenue most? (2) How do channels overlap? "
        "(3) What patterns matter for budget decisions? Be concise but insightful."
    )
    
    user_prompt = f"""
    {eda_prompt}
    
    Generate a 2-3 paragraph narrative explaining these EDA findings for {segment_name} level.
    Highlight the most important insights and actionable recommendations.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # Updated model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating narrative: {str(e)}]"


def generate_outlier_rationale(
    client: AzureOpenAI,
    outlier_details: Dict[str, Any],
    max_tokens: int = 600
) -> str:
    """
    Generate narrative explaining outlier removal decisions.
    
    Args:
        client: AzureOpenAI client
        outlier_details: {
            "removed_indices": [list of month/row indices],
            "reasons": ["reason1", "reason2"],
            "affected_columns": ["col1", "col2"],
            "before_after_stats": {col: {before: stats, after: stats}}
        }
        max_tokens: Response length
        
    Returns:
        str: Narrative text
    """
    system_prompt = (
        "You are a data scientist explaining data cleaning decisions to non-technical stakeholders. "
        "Explain WHY outliers were removed in business terms. "
        "Focus on: (1) What was the anomaly? (2) Why is it a problem? (3) What's the impact?"
    )
    
    user_prompt = f"""
    Outliers removed: {len(outlier_details.get('removed_indices', []))}
    Reasons: {outlier_details.get('reasons', [])}
    Affected columns: {outlier_details.get('affected_columns', [])}
    
    Generate a 1-2 paragraph explanation of these outlier removals.
    Be specific about the business rationale.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating narrative: {str(e)}]"


def generate_feature_engineering_narrative(
    client: AzureOpenAI,
    feature_decisions: Dict[str, Any],
    max_tokens: int = 700
) -> str:
    """
    Generate narrative explaining feature engineering choices.
    
    Args:
        client: AzureOpenAI client
        feature_decisions: {
            "transformations": {"channel": "log", ...},
            "combined_channels": [["channel1", "channel2"], ...],
            "multicollinearity": {"VIF": {...}},
            "reasoning": [...]
        }
        max_tokens: Response length
        
    Returns:
        str: Narrative text
    """
    system_prompt = (
        "You are a marketing data scientist explaining feature engineering decisions. "
        "Explain transformations and channel combinations in business and statistical terms. "
        "Focus on: (1) What transformation was applied and why? (2) Which channels were combined and why? "
        "(3) How does this improve model quality?"
    )
    
    user_prompt = f"""
    Transformations: {feature_decisions.get('transformations', {})}
    Combined channels: {feature_decisions.get('combined_channels', [])}
    Multicollinearity status: {feature_decisions.get('multicollinearity', {})}
    
    Generate a 2-3 paragraph narrative explaining these feature engineering decisions.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating narrative: {str(e)}]"


def generate_model_ranking_narrative(
    client: AzureOpenAI,
    ranking_prompt: str,
    max_tokens: int = 1000
) -> str:
    """
    Generate narrative for top-ranked models.
    
    Args:
        client: AzureOpenAI client
        ranking_prompt: Formatted model ranking data (from modeling_enhanced.prepare_model_ranking_narrative)
        max_tokens: Response length
        
    Returns:
        str: Narrative text
    """
    system_prompt = (
        "You are a marketing mix modeling expert. Your task is to explain model rankings "
        "and recommend the best model. Focus on: (1) Why these models ranked high? "
        "(2) What are their strengths? (3) Which is best for decision-making? "
        "(4) Any caveats or limitations?"
    )
    
    user_prompt = f"""
    {ranking_prompt}
    
    Generate a 3-4 paragraph narrative:
    1. Explain the top 3 models and why they ranked highest
    2. Compare their strengths and weaknesses
    3. Recommend the single best model for business decisions
    4. Highlight any ordinality or stability concerns
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating narrative: {str(e)}]"


def generate_optimization_narrative(
    client: AzureOpenAI,
    scenario_data: Dict[str, Any],
    max_tokens: int = 1200
) -> str:
    """
    Generate narrative comparing optimization scenarios.
    
    Args:
        client: AzureOpenAI client
        scenario_data: {
            "base_case": {...},
            "budget_neutral": {...},
            "max_profit": {...},
            "blue_sky": {...}
        }
        max_tokens: Response length
        
    Returns:
        str: Narrative text
    """
    system_prompt = (
        "You are a marketing mix optimization expert. Explain budget allocation scenarios "
        "for marketing executives. Focus on: (1) What's each scenario's goal? "
        "(2) How does budget shift between channels? (3) What's the profit uplift? "
        "(4) Which scenario do you recommend and why?"
    )
    
    user_prompt = f"""
    Optimization Scenarios:
    {json.dumps(scenario_data, indent=2)}
    
    Generate a 4-5 paragraph narrative:
    1. Summarize each scenario's strategy and expected ROI
    2. Compare channel allocations across scenarios
    3. Identify winners and losers (channels that gain/lose budget)
    4. Quantify profit implications
    5. Recommend best scenario with caveats
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating narrative: {str(e)}]"


# =============================================================================
# GENERIC NARRATIVE GENERATOR (Fallback)
# =============================================================================

def generate_narrative(
    client: AzureOpenAI,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.3
) -> str:
    """
    Generic narrative generator for custom prompts.
    
    Args:
        client: AzureOpenAI client
        prompt: User prompt / data to explain
        system_prompt: Optional system context
        max_tokens: Response length
        temperature: Creativity (0 = deterministic, 1 = creative)
        
    Returns:
        str: Generated narrative
    """
    if system_prompt is None:
        system_prompt = (
            "You are a senior marketing analytics consultant. "
            "Explain data findings in clear, actionable business language. "
            "Be concise but insightful. Always explain: what happened, why it matters, what to do about it."
        )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating narrative: {str(e)}]"


# =============================================================================
# AGENT CRITIQUE & DECISION-MAKING
# =============================================================================

def get_agent_critique(
    client: AzureOpenAI,
    analysis_summary: Dict[str, Any],
    max_tokens: int = 800
) -> Dict[str, Any]:
    """
    Get LLM critique of analysis quality and recommendations for improvement.
    
    Args:
        client: AzureOpenAI client
        analysis_summary: {
            "model_fit": float (R²),
            "ordinality_violations": int,
            "multicollinearity": float (max VIF),
            "data_quality": str,
            ...
        }
        max_tokens: Response length
        
    Returns:
        {
            "critique": str,
            "recommendations": [str, ...],
            "next_steps": [str, ...]
        }
    """
    system_prompt = (
        "You are a statistical quality assurance expert for marketing mix modeling. "
        "Critique the current analysis and recommend improvements to ensure robustness and reliability."
    )
    
    user_prompt = f"""
    Current Analysis Status:
    {json.dumps(analysis_summary, indent=2)}
    
    Provide:
    1. A 2-3 sentence critique of analysis quality
    2. 3-4 specific recommendations for improvement
    3. 2-3 concrete next steps (e.g., 'Remove column X', 'Re-fit with transformation Y')
    
    Focus on statistical rigor and business relevance.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        # Parse response into structured format
        critique_text = response.choices[0].message.content
        
        return {
            "critique": critique_text,
            "recommendations": [],  # Could parse further if needed
            "next_steps": []
        }
    except Exception as e:
        return {
            "critique": f"[Error: {str(e)}]",
            "recommendations": [],
            "next_steps": []
        }
