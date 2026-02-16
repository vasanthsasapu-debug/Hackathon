"""
Export Package - Optimization & Delivery (Excel, PowerPoint, Goals)

Modules:
  - optimizer: 4-scenario mix optimization (Base Case, Budget Neutral, Max Profit, Blue Sky)
  - excel: Multi-sheet Excel export (EDA, Models, Response Curves, Optimization, Narratives)
  - powerpoint: Professional 10-slide PowerPoint presentation
  - goals: NLP goal parsing and constraint-based optimization
"""

from .optimizer import run_mix_optimization
from .excel import export_to_excel
from .powerpoint import generate_ppt_presentation
from .goals import GoalParser, run_goal_based_optimization

__all__ = [
    "run_mix_optimization",
    "export_to_excel",
    "generate_ppt_presentation",
    "GoalParser",
    "run_goal_based_optimization",
]
