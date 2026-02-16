"""
=============================================================================
GOAL PARSING ENGINE
=============================================================================
Parse natural language constraints into structured optimization goals.

Pattern-matching approach, no LLM required.
"""

import re
from typing import Dict, List, Tuple, Any, Optional


# =============================================================================
# CONSTRAINT PATTERNS
# =============================================================================

CHANNEL_ALIASES = {
    'email': ['email', 'emailing', 'e-mail'],
    'search': ['search', 'google', 'sem', 'ppc', 'adwords'],
    'social': ['social', 'facebook', 'instagram', 'twitter', 'tiktok', 'linkedin'],
    'display': ['display', 'banner', 'remarketing', 'programmatic'],
    'affiliate': ['affiliate', 'partner', 'commission'],
    'organic': ['organic', 'seo', 'natural'],
    'direct': ['direct', 'brand'],
    'tv': ['tv', 'television', 'broadcast'],
    'radio': ['radio'],
    'print': ['print', 'magazine', 'newspaper'],
}

# Reverse mapping for fast lookup
CHANNEL_LOOKUP = {}
for channel, aliases in CHANNEL_ALIASES.items():
    for alias in aliases:
        CHANNEL_LOOKUP[alias.lower()] = channel


# =============================================================================
# METRIC EXTRACTION
# =============================================================================

def extract_percentage(text: str) -> Optional[float]:
    """Extract percentage value from text."""
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text.lower())
    if matches:
        return float(matches[0]) / 100
    return None


def extract_absolute_value(text: str) -> Optional[float]:
    """Extract absolute value (e.g., '$1000', '1000')."""
    matches = re.findall(r'\$?(\d+(?:[,\.]\d+)*(?:\.\d+)?)', text.lower())
    if matches:
        # Remove commas and parse
        value_str = matches[0].replace(',', '')
        try:
            return float(value_str)
        except:
            return None
    return None


def extract_metric(text: str) -> Optional[str]:
    """
    Extract metric type: 'roi', 'roas', 'sales', 'spend', 'profit', 'reach', 'frequency', etc.
    """
    text_lower = text.lower()
    
    if 'roi' in text_lower:
        return 'roi'
    elif 'roas' in text_lower:
        return 'roas'
    elif 'profit' in text_lower or 'margin' in text_lower:
        return 'profit'
    elif 'sale' in text_lower or 'revenue' in text_lower:
        return 'sales'
    elif 'spend' in text_lower or 'budget' in text_lower:
        return 'spend'
    elif 'reach' in text_lower:
        return 'reach'
    elif 'frequency' in text_lower:
        return 'frequency'
    elif 'conversion' in text_lower or 'ctr' in text_lower:
        return 'conversion'
    
    return None


def extract_operator(text: str) -> Optional[str]:
    """Extract comparison operator: 'increase', 'decrease', 'maintain', 'reach', 'minimize', 'maximize'."""
    text_lower = text.lower()
    
    if 'increase' in text_lower or 'grow' in text_lower or 'boost' in text_lower or 'improve' in text_lower:
        return 'increase'
    elif 'decrease' in text_lower or 'reduce' in text_lower or 'cut' in text_lower or 'lower' in text_lower:
        return 'decrease'
    elif 'maintain' in text_lower or 'keep' in text_lower or 'stay' in text_lower:
        return 'maintain'
    elif 'reach' in text_lower or 'target' in text_lower or 'achieve' in text_lower:
        return 'reach'
    elif 'minimize' in text_lower:
        return 'minimize'
    elif 'maximize' in text_lower:
        return 'maximize'
    
    return None


def extract_channel(text: str) -> Optional[str]:
    """Extract channel name from text."""
    text_lower = text.lower()
    
    # Check against known channels
    words = re.findall(r'\b\w+\b', text_lower)
    
    for word in words:
        if word in CHANNEL_LOOKUP:
            return CHANNEL_LOOKUP[word]
    
    return None


# =============================================================================
# CONSTRAINT PARSERS
# =============================================================================

def parse_channel_constraint(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse channel-level constraint.
    
    Examples:
      "increase email spend by 20%"
      "reduce social by 15%"
      "maximize search roas"
      
    Returns:
        {
            'type': 'channel_constraint',
            'channel': 'email',
            'metric': 'spend',
            'operator': 'increase',
            'value': 0.20,
        }
    """
    channel = extract_channel(text)
    if not channel:
        return None
    
    metric = extract_metric(text)
    operator = extract_operator(text)
    value = extract_percentage(text) or extract_absolute_value(text)
    
    if operator and metric:
        return {
            'type': 'channel_constraint',
            'channel': channel,
            'metric': metric or 'spend',
            'operator': operator,
            'value': value,
        }
    
    return None


def parse_portfolio_constraint(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse portfolio-level constraint.
    
    Examples:
      "maintain overall roi at 50%"
      "maximize profit"
      "minimize total spend"
      "reach 100k in sales"
      
    Returns:
        {
            'type': 'portfolio_constraint',
            'metric': 'roi',
            'operator': 'maintain',
            'value': 0.50,
        }
    """
    metric = extract_metric(text)
    operator = extract_operator(text)
    value = extract_percentage(text) or extract_absolute_value(text)
    
    # Portfolio constraints: no specific channel
    if metric and operator:
        return {
            'type': 'portfolio_constraint',
            'metric': metric,
            'operator': operator,
            'value': value,
        }
    
    return None


def parse_budget_constraint(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse budget constraint.
    
    Examples:
      "allocate $50k to search"
      "spend max 30k on social"
      "invest at least 5k in email"
    """
    # Budget keywords
    if 'allocate' not in text.lower() and 'spend' not in text.lower() and 'invest' not in text.lower():
        return None
    
    channel = extract_channel(text)
    value = extract_absolute_value(text)
    
    if 'max' in text.lower() or 'at most' in text.lower() or 'no more than' in text.lower():
        operator = 'max'
    elif 'min' in text.lower() or 'at least' in text.lower() or 'minimum' in text.lower():
        operator = 'min'
    else:
        operator = 'allocate'
    
    if value:
        return {
            'type': 'budget_constraint',
            'channel': channel,
            'operator': operator,
            'value': value,
        }
    
    return None


def parse_constraint_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single constraint line.
    
    Tries: budget → channel → portfolio
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    # Try budget first
    result = parse_budget_constraint(line)
    if result:
        return result
    
    # Try channel constraint
    result = parse_channel_constraint(line)
    if result:
        return result
    
    # Try portfolio constraint
    result = parse_portfolio_constraint(line)
    if result:
        return result
    
    return None


# =============================================================================
# ORCHESTRATION
# =============================================================================

def parse_goals(goals_text: str) -> Dict[str, Any]:
    """
    Parse goals/constraints from natural language text.
    
    Args:
        goals_text: Multiline text with constraints
        
    Returns:
        {
            'channel_constraints': [...],
            'portfolio_constraints': [...],
            'budget_constraints': [...],
            'log': [...],
            'is_valid': bool,
        }
    """
    channel_constraints = []
    portfolio_constraints = []
    budget_constraints = []
    log = []
    
    lines = goals_text.split('\n')
    
    for line in lines:
        constraint = parse_constraint_line(line)
        
        if constraint:
            if constraint['type'] == 'channel_constraint':
                channel_constraints.append(constraint)
                log.append(f"✅ Channel constraint: {constraint['channel']} {constraint['operator']} {constraint['metric']} by {constraint.get('value', 'TBD')}")
            
            elif constraint['type'] == 'portfolio_constraint':
                portfolio_constraints.append(constraint)
                log.append(f"✅ Portfolio constraint: {constraint['operator']} {constraint['metric']} {constraint.get('value', '')}")
            
            elif constraint['type'] == 'budget_constraint':
                budget_constraints.append(constraint)
                channel_name = constraint.get('channel', 'unspecified')
                log.append(f"✅ Budget constraint: {channel_name} {constraint['operator']} ${constraint.get('value', 0)}")
        
        elif line.strip() and not line.strip().startswith('#'):
            log.append(f"⚠️  Could not parse: '{line}'")
    
    return {
        'channel_constraints': channel_constraints,
        'portfolio_constraints': portfolio_constraints,
        'budget_constraints': budget_constraints,
        'log': log,
        'is_valid': len(channel_constraints) > 0 or len(portfolio_constraints) > 0 or len(budget_constraints) > 0,
    }


def convert_constraints_to_optimization_params(
    constraints: Dict[str, Any],
    current_allocation: Dict[str, float],
    channel_columns: List[str],
) -> Dict[str, Any]:
    """
    Convert parsed constraints into optimization function parameters.
    
    Args:
        constraints: Output from parse_goals()
        current_allocation: Current spend by channel
        channel_columns: List of channel names
        
    Returns:
        {
            'budget_constraints': {...},
            'channel_constraints': {...},
            'portfolio_constraints': {...},
        }
    """
    budget_dict = {}
    channel_dict = {}
    portfolio_dict = {}
    
    # Process budget constraints
    for constraint in constraints.get('budget_constraints', []):
        channel = constraint.get('channel')
        value = constraint.get('value')
        operator = constraint.get('operator')
        
        if channel and value:
            if operator == 'allocate':
                budget_dict[channel] = {'target': value}
            elif operator == 'max':
                budget_dict[channel] = {'max': value}
            elif operator == 'min':
                budget_dict[channel] = {'min': value}
    
    # Process channel constraints
    for constraint in constraints.get('channel_constraints', []):
        channel = constraint.get('channel')
        metric = constraint.get('metric')
        operator = constraint.get('operator')
        value = constraint.get('value')
        
        if channel and metric:
            if channel not in channel_dict:
                channel_dict[channel] = {}
            
            channel_dict[channel][metric] = {
                'operator': operator,
                'value': value,
            }
    
    # Process portfolio constraints
    for constraint in constraints.get('portfolio_constraints', []):
        metric = constraint.get('metric')
        operator = constraint.get('operator')
        value = constraint.get('value')
        
        if metric:
            portfolio_dict[metric] = {
                'operator': operator,
                'value': value,
            }
    
    return {
        'budget_constraints': budget_dict,
        'channel_constraints': channel_dict,
        'portfolio_constraints': portfolio_dict,
    }


# =============================================================================
# UTILITIES
# =============================================================================

def generate_sample_goal_template() -> str:
    """Generate a sample goals text for user reference."""
    return """
# Marketing Spend Optimization Goals

## Channel-Level Goals
increase email spend by 20%
reduce social budget by 15%
maintain search at current level

## Portfolio Goals
maximize overall roi
maintain profit at 40%
reach 100k in total sales

## Budget Constraints
allocate $50k to email
spend max 30k on social
invest at least 5k in search
""".strip()
