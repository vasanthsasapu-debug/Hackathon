"""
=============================================================================
ENGINE EXPORT -- Thin wrappers for output generation
=============================================================================
Modules:
  - excel: Export results to Excel workbook
  - powerpoint: Export results to PowerPoint presentation

These are thin wrappers that:
  1. Call engine modules (eda_metrics, modeling, etc.) for computation
  2. Format results for output
  3. Write to files

No computation logic here - all compute is in parent engine modules.
"""

from . import excel
from . import powerpoint

__all__ = [
    "excel",
    "powerpoint",
]
