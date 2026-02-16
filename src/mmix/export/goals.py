"""
=============================================================================
GOAL-BASED OPTIMIZER -- NLP-based goal parsing & optimization
=============================================================================
Parses natural language goals and converts to optimization constraints:
  - "Increase Email channel ROI by 2%" → constraint for optimization
  - "Keep TV spend below $50M" → bound constraint
  - "Maximize Digital ROI" → objective function weight
=============================================================================
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# NLP GOAL PARSER
# =============================================================================

class GoalParser:
    """Parse natural language goals into optimization constraints."""
    
    # Pattern matching for common goal types
    PATTERNS = {
        "roi_increase": r"increase\s+(\w+)\s+(?:roi|roas)\s+by\s+([\d.]+)\s*%",
        "channel_minimum": r"keep\s+(\w+)\s+spend\s+(?:above|at least)\s*\$?([\d.]+)(?:m|million)?",
        "channel_maximum": r"keep\s+(\w+)\s+spend\s+(?:below|under)\s*\$?([\d.]+)(?:m|million)?",
        "maximize_channel": r"maximize\s+(\w+)(?:\s+roi|\s+roas)?",
        "minimize_channel": r"minimize\s+(\w+)(?:\s+spend)?",
        "total_budget": r"total\s+(?:budget|spend)\s+(?:of|is|equals?)\s*\$?([\d.]+)(?:m|million)?",
        "channel_ratio": r"(\w+)\s+should\s+be\s+(?:at least|no more than)\s+([\d.]+)%\s+of\s+budget",
    }
    
    def __init__(self):
        """Initialize parser."""
        pass
    
    def parse(self, goal_text: str) -> Dict[str, Any]:
        """
        Parse a goal string into structured constraints.
        
        Args:
            goal_text: Natural language goal (e.g., "increase Email ROI by 2%")
            
        Returns:
            {
                "goal_type": str,
                "channel": str,
                "metric": str,
                "target_value": float,
                "original_text": str,
                "confidence": 0-1
            }
        """
        goal_text_lower = goal_text.lower().strip()
        
        # Try to match against patterns
        for goal_type, pattern in self.PATTERNS.items():
            match = re.search(pattern, goal_text_lower)
            if match:
                return {
                    "goal_type": goal_type,
                    "groups": match.groups(),
                    "original_text": goal_text,
                    "confidence": 0.9,
                }
        
        # No match found
        return {
            "goal_type": "unknown",
            "original_text": goal_text,
            "confidence": 0.0,
        }
    
    def parse_multiple(self, goal_texts: List[str]) -> List[Dict[str, Any]]:
        """Parse multiple goals."""
        return [self.parse(goal) for goal in goal_texts]


# =============================================================================
# CONSTRAINT BUILDERS
# =============================================================================

def build_roi_increase_constraint(
    channel: str,
    current_roi: float,
    increase_pct: float,
    elasticities: Dict[str, float]
) -> Dict[str, Any]:
    """
    Build constraint for "increase {channel} ROI by {pct}%".
    
    ROI_{new} = ROI_{current} * (1 + increase_pct/100)
    
    Args:
        channel: Channel name
        current_roi: Current ROI for this channel
        increase_pct: Percentage increase (e.g., 2 for 2%)
        elasticities: Channel elasticities
        
    Returns:
        Constraint dict for optimizer
    """
    target_roi = current_roi * (1 + increase_pct / 100)
    
    return {
        "constraint_type": "roi_increase",
        "channel": channel,
        "current_roi": current_roi,
        "target_roi": target_roi,
        "increase_pct": increase_pct,
        "elasticity": elasticities.get(channel, 0),
    }


def build_spend_bound_constraint(
    channel: str,
    bound_type: str,  # "min" or "max"
    bound_value: float
) -> Dict[str, Any]:
    """
    Build constraint for "keep {channel} spend {above/below} ${value}".
    
    Args:
        channel: Channel name
        bound_type: "minimum" or "maximum"
        bound_value: Spend limit
        
    Returns:
        Constraint dict
    """
    return {
        "constraint_type": "spend_bound",
        "channel": channel,
        "bound_type": bound_type,
        "bound_value": bound_value,
    }


def build_channel_ratio_constraint(
    channel: str,
    ratio_type: str,  # "min" or "max"
    ratio_pct: float,
    total_budget: float
) -> Dict[str, Any]:
    """
    Build constraint for "{channel} should be {min/max} {pct}% of budget".
    
    Args:
        channel: Channel name
        ratio_type: "minimum" or "maximum"
        ratio_pct: Percentage of budget
        total_budget: Total budget
        
    Returns:
        Constraint dict
    """
    return {
        "constraint_type": "channel_ratio",
        "channel": channel,
        "ratio_type": ratio_type,
        "ratio_pct": ratio_pct,
        "budget_bound": (ratio_pct / 100) * total_budget,
    }


# =============================================================================
# CONSTRAINT APPLICATION
# =============================================================================

def apply_constraints_to_optimization(
    constraints: List[Dict[str, Any]],
    base_allocation: Dict[str, float],
    total_budget: float
) -> Dict[str, Any]:
    """
    Apply parsed constraints to modify optimization.
    
    Args:
        constraints: List of constraint dicts
        base_allocation: Current channel allocation
        total_budget: Total budget constraint
        
    Returns:
        {
            "adjusted_bounds": {channel: (min, max)},
            "objective_weights": {channel: weight},
            "applied_constraints": [...],
            "violations": [...]
        }
    """
    adjusted_bounds = {}
    objective_weights = {}
    applied = []
    violations = []
    
    # Initialize bounds from base allocation ± 50%
    for channel, spend in base_allocation.items():
        adjusted_bounds[channel] = (spend * 0.5, spend * 1.5)
        objective_weights[channel] = 1.0
    
    # Apply constraints
    for constraint in constraints:
        constraint_type = constraint.get("constraint_type")
        
        if constraint_type == "spend_bound":
            channel = constraint["channel"]
            bound_type = constraint["bound_type"]
            bound_value = constraint["bound_value"]
            
            current_min, current_max = adjusted_bounds.get(channel, (0, total_budget))
            
            if bound_type == "minimum":
                adjusted_bounds[channel] = (max(current_min, bound_value), current_max)
                applied.append(f"Set minimum spend for {channel} to ${bound_value:.0f}")
            else:  # maximum
                adjusted_bounds[channel] = (current_min, min(current_max, bound_value))
                applied.append(f"Set maximum spend for {channel} to ${bound_value:.0f}")
        
        elif constraint_type == "roi_increase":
            channel = constraint["channel"]
            target_roi = constraint["target_roi"]
            elasticity = constraint["elasticity"]
            
            # Weight this channel higher in optimization
            if elasticity > 0:
                objective_weights[channel] = 1.5  # Prioritize this channel
                applied.append(f"Prioritized {channel} to achieve {constraint['increase_pct']:.1f}% ROI increase")
            else:
                violations.append(f"Cannot increase {channel} ROI: negative/zero elasticity")
        
        elif constraint_type == "channel_ratio":
            channel = constraint["channel"]
            ratio_type = constraint["ratio_type"]
            budget_bound = constraint["budget_bound"]
            
            current_min, current_max = adjusted_bounds.get(channel, (0, total_budget))
            
            if ratio_type == "minimum":
                adjusted_bounds[channel] = (max(current_min, budget_bound), current_max)
                applied.append(f"Set {channel} to at least {constraint['ratio_pct']:.1f}% of budget")
            else:  # maximum
                adjusted_bounds[channel] = (current_min, min(current_max, budget_bound))
                applied.append(f"Cap {channel} at {constraint['ratio_pct']:.1f}% of budget")
    
    return {
        "adjusted_bounds": adjusted_bounds,
        "objective_weights": objective_weights,
        "applied_constraints": applied,
        "violations": violations,
    }


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_goal_based_optimization(
    goal_text: str,
    base_allocation: Dict[str, float],
    current_metrics: Dict[str, float],
    elasticities: Dict[str, float],
    total_budget: float = None
) -> Dict[str, Any]:
    """
    Main entry point: parse goal and generate optimization constraints.
    
    Args:
        goal_text: Natural language goal (single or comma-separated)
        base_allocation: Current channel allocation
        current_metrics: {channel: current_roi} or similar
        elasticities: Channel elasticities
        total_budget: Total budget (derived from base_allocation if None)
        
    Returns:
        {
            "parsed_goals": [...],
            "constraints": [...],
            "applied_constraints": [...],
            "violations": [...],
            "log": [...]
        }
    """
    if total_budget is None:
        total_budget = sum(base_allocation.values())
    
    log = []
    
    # Parse goals
    parser = GoalParser()
    goal_list = [g.strip() for g in goal_text.split(",")]
    parsed_goals = parser.parse_multiple(goal_list)
    
    log.append(f"Parsed {len([g for g in parsed_goals if g['confidence'] > 0])} goals")
    
    # Build constraints
    constraints = []
    
    for parsed_goal in parsed_goals:
        if parsed_goal["confidence"] == 0:
            log.append(f"⚠️  Could not parse goal: {parsed_goal['original_text']}")
            continue
        
        goal_type = parsed_goal["goal_type"]
        groups = parsed_goal.get("groups", ())
        
        try:
            if goal_type == "roi_increase":
                channel, increase_pct = groups
                increase_pct = float(increase_pct)
                current_roi = current_metrics.get(channel, 1.0)
                constraint = build_roi_increase_constraint(channel, current_roi, increase_pct, elasticities)
                constraints.append(constraint)
                log.append(f"✅ Goal: Increase {channel} ROI by {increase_pct}%")
            
            elif goal_type == "channel_maximum":
                channel, amount = groups
                amount = float(amount) * 1e6 if "million" in goal_type else float(amount)
                constraint = build_spend_bound_constraint(channel, "maximum", amount)
                constraints.append(constraint)
                log.append(f"✅ Goal: Cap {channel} spend at ${amount:.0f}")
            
            elif goal_type == "channel_minimum":
                channel, amount = groups
                amount = float(amount) * 1e6 if "million" in goal_type else float(amount)
                constraint = build_spend_bound_constraint(channel, "minimum", amount)
                constraints.append(constraint)
                log.append(f"✅ Goal: Minimum {channel} spend of ${amount:.0f}")
            
            elif goal_type == "total_budget":
                new_budget = float(groups[0]) * 1e6 if "million" in goal_type else float(groups[0])
                total_budget = new_budget
                log.append(f"✅ Goal: Total budget set to ${new_budget:.0f}")
        
        except Exception as e:
            log.append(f"❌ Error processing goal {parsed_goal['original_text']}: {str(e)}")
    
    # Apply constraints
    application_result = apply_constraints_to_optimization(constraints, base_allocation, total_budget)
    
    return {
        "parsed_goals": parsed_goals,
        "constraints": constraints,
        "applied_constraints": application_result["applied_constraints"],
        "adjusted_bounds": application_result["adjusted_bounds"],
        "objective_weights": application_result["objective_weights"],
        "violations": application_result["violations"],
        "log": log,
    }
