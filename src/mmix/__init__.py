"""
MMIX Package - Production-Ready Marketing Mix Modeling Platform

Components:
  - pipeline: Core EDA, modeling, and response curve extraction
  - agents: Agentic orchestration with LLM integration
  - export: Optimization, Excel/PowerPoint export, goal-based constraints
  - legacy: Archive of older implementations
  - config: Centralized configuration (Azure, hyperparameters)
"""

__version__ = "1.0.0"
__author__ = "MMIX Team"

from . import pipeline
from . import agents
from . import export
from . import config

__all__ = ["pipeline", "agents", "export", "config"]
