"""
Agents Package - Agentic Orchestration & LLM Integration

Modules:
  - orchestrator: State machine pipeline orchestration with feedback loops
  - llm: Azure OpenAI integration for narrative generation
"""

from .orchestrator import Orchestrator, PipelineState
from .llm import (
    get_llm_client,
    generate_eda_narrative,
    generate_outlier_rationale,
    generate_feature_engineering_narrative,
    generate_model_ranking_narrative,
    generate_optimization_narrative,
    get_agent_critique,
)

__all__ = [
    "Orchestrator",
    "PipelineState",
    "get_llm_client",
    "generate_eda_narrative",
    "generate_outlier_rationale",
    "generate_feature_engineering_narrative",
    "generate_model_ranking_narrative",
    "generate_optimization_narrative",
    "get_agent_critique",
]
