# Workflow: Mix Optimization

## TLDR
Apply response curves to create 4 budget scenarios: Base Case (historical), Budget Neutral (within-channel only), Max Profit (cross-channel), Blue Sky (unconstrained) → Recommend allocation → Compare outcomes.

---

## Summary

### Objective
Generate actionable budget recommendations across marketing channels to maximize profit or ROI.

### Inputs
- Response curves (from response_curves workflow)
- Historical budget allocation
- Constraints: budget ceiling, minimum spend per channel, fixed costs

### Engine Module
`src/mmix/engine/optimization_engine.py`

### Steps
1. **Base Case Scenario**
   - Use historical spend per channel
   - Predict revenue using response curves
   - Compute profit (revenue - spend)
   - Baseline for comparison

2. **Budget Neutral Scenario**
   - Constraint: Total spend = historical total
   - Optimize: Reallocate budget within channel (e.g., move digital budget from Email to SEM)
   - Objective: Maximize profit within each channel
   - Return: New allocation per channel-segment combo

3. **Max Profit Scenario**
   - Constraint: Total spend = historical total, but channels can change
   - Optimize: Cross-channel reallocation
   - Objective: Maximize profit globally
   - Return: New allocation per channel-segment combo

4. **Blue Sky Scenario**
   - Constraint: None (or user-defined)
   - Optimize: Unconstrained budget for each channel
   - Objective: Maximize profit (mROI = 0%)
   - Return: Ideal allocation (for comparison/business validation)

5. **Constraint Handling** (user-defined via prompt or config)
   - Min spend per channel: "TV >= $100K"
   - Max spend per channel: "Digital <= $500K"
   - Elasticity constraints: "preserve ordinality"

### Outputs
```json
{
  "scenarios": {
    "base_case": {
      "allocation": {"TV": 1000, "Digital": 500, ...},
      "predicted_revenue": 50000,
      "profit": 40000,
      "mROI": 40.0
    },
    "budget_neutral": {
      "allocation": {"TV": 950, "Digital": 550, ...},
      "predicted_revenue": 50500,
      "profit": 40500,
      "mROI": 41.5,
      "improvement": 500
    },
    "max_profit": {
      "allocation": {"TV": 800, "Digital": 700, ...},
      "predicted_revenue": 51500,
      "profit": 41500,
      "mROI": 42.3,
      "improvement": 1500
    },
    "blue_sky": {
      "allocation": {"TV": 500, "Digital": 1200, ...},
      "predicted_revenue": 53000,
      "profit": 43000,
      "mROI": 43.0,
      "improvement": 3000
    }
  },
  "recommendations": {
    "best_scenario": "max_profit",
    "key_moves": [
      "Reduce TV by 20% (diminishing returns evident)",
      "Increase Digital by 40% (highest elasticity)"
    ]
  }
}
```

### Validation Rules
- ✅ All scenarios have non-negative allocation per channel
- ✅ Total spend respects constraint (or marked as unconstrained)
- ✅ Improvements monotonic: Base ≤ Budget Neutral ≤ Max Profit ≤ Blue Sky
- ⚠️ Warn if improvement > 25% (verify elasticities; unusually high ROI)
- ❌ Fail if optimization does not converge (solver issue)

### Edge Cases
- **Zero elasticity channel** → Optimizer may eliminate entirely; allow min-spend constraint
- **User constraint conflicts with data** (e.g., "TV <= $100 but historical is $500") → Warn, proceed with constraint
- **Multiple equal-profit solutions** → Return all, let user choose (simplicity, risk)
- **Solver timeout** → Return best-found solution, flag as incomplete

### Performance
- Optimization (scipy.optimize.minimize): <5 seconds per scenario
- All 4 scenarios: ~20 seconds

### Future Vision (V2)
- Goal-based formulation: Parse user prompt ("increase Email ROI by 2%") → Auto-formulate constraint
- LLM explanation: "We recommend moving $50K from TV to Digital because Digital has higher elasticity and TV shows diminishing returns beyond $800K"
- Scenario sensitivity analysis: "Max Profit improves by $100K if we allow 5% TV reduction"
