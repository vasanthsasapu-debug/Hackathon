# Workflow: Response Curves & Elasticity

## TLDR
Use selected model to compute predicted sales across budget ranges per channel → Fit response curves → Calculate elasticity (% change in sales per 1% budget increase) → Return elasticities + visualizations.

---

## Summary

### Objective
Quantify channel ROI and diminishing returns for optimization.

### Inputs
- Best model (from modeling workflow)
- Engineered DataFrame
- Budget ranges to test (e.g., 50% to 150% of historical spend)

### Engine Module
`src/mmix/engine/response_curves.py`

### Steps
1. **Define Budget Grid** → For each channel, create array of spend values (e.g., [0.5×historical, ..., 1.5×historical])
2. **Predict Sales** → Use best model to predict sales for each budget combination
3. **Fit Response Curves** → Fit linear/log/power-law to (budget, predicted_sales) pairs
4. **Compute Elasticity** → Slope of log-log curve: d(log Sales) / d(log Budget)
5. **Confidence Bands** → Bootstrap or analytical confidence intervals on elasticities
6. **Per-Segment Curves** (if applicable) → Repeat for each segment

### Outputs
```json
{
  "response_curves": {
    "TV": {
      "curve_type": "power-law",
      "formula": "sales = 1000 * (budget)^0.65",
      "elasticity": 0.65,
      "elasticity_ci": [0.60, 0.70],
      "points": [[100, 1000], [120, 1090], ...]
    },
    ...
  },
  "elasticity_ranking": [
    {"channel": "TV", "elasticity": 0.65},
    {"channel": "Digital", "elasticity": 0.58},
    ...
  ]
}
```

### Validation Rules
- ✅ Elasticity in (0, 2) for most marketing channels (diminishing returns)
- ✅ Elasticity > 0 (positive relationship with sales)
- ⚠️ Warn if elasticity > 1.5 (increasing returns; unusual, check data)
- ❌ Fail if elasticity < 0 (negative relationship; model/data error)

### Edge Cases
- **Budget range outside historical data** → Extrapolation; flag confidence as lower
- **Flat response curve (elasticity ≈ 0)** → Channel has minimal impact; consider removing
- **Non-monotonic predicted sales** → Suggest different curve fit or model re-evaluation
- **Segment with too few observations** → Skip segment, flag in logs

### Performance
- Response curve computation: <5 seconds for 50 budget points × 10 channels

### Visualization
- X-axis: Budget (% of historical), Y-axis: Predicted Sales
- Show actual historical point, response curve, confidence bands
- Include elasticity label per curve
