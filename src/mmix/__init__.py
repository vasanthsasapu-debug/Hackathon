"""
MMIX Package - Production-Ready Marketing Mix Modeling Platform

Components:
  - pipeline: Core EDA, modeling, and response curve extraction
  - export: Optimization, Excel/PowerPoint export, goal-based constraints
  - legacy: Archive of older implementations
  - config: Centralized configuration (Azure, hyperparameters)

Note: agents and engine are now at project root:
  - /agents/: Agentic orchestration with LLM integration
  - /engine/: Deterministic computation layer
"""

__version__ = "1.0.0"
__author__ = "MMIX Team"

from . import pipeline
from . import export
from . import config

__all__ = ["pipeline", "export", "config"]
