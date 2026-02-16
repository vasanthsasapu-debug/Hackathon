# Workflow: Feature Engineering & Transformation

## TLDR
Auto-select response curve shapes (linear, log, power-law, diminishing-return) per channel → Apply transformations → Check ordinality → Flag multicollinearity → Return transformed features.

---

## Summary

### Objective
Optimize feature representation for modeling: linearize non-linear relationships, preserve ordinality for key channels, handle collinearity.

### Inputs
- Cleaned DataFrame (from outlier_removal workflow)
- Column classification dict
- Feature config: curves_to_test, ordinality_channels, vif_threshold

### Engine Module
`src/mmix/engine/response_curves.py` (fit_response_curve) + `src/mmix/engine/utilities.py` (check_multicollinearity)

### Steps
1. **Response Curve Fitting** → For each Promotional_Activity channel, fit all curves (linear, log, log-log, power-law, diminishing-return) and select best via AIC/BIC
2. **Apply Transformation** → Transform channel values using selected curve
3. **Ordinality Check** → For key channels (e.g., TV, SEM), ensure monotonic relationship with Sales_Output
4. **Multicollinearity Detection** → Compute VIF; flag if > threshold (default 5)
5. **Feature Combination** → If multicollinearity high, suggest combining similar channels (e.g., TV + Radio → "Broadcast")
6. **Return** → Engineered DataFrame + transformation metadata

### Outputs
```json
{
  "engineered_df": DataFrame,
  "transformations": {
    "TV": {"curve": "log", "formula": "log(TV + 1)", "r2": 0.82},
    "Digital": {"curve": "power-law", "formula": "(Digital)^0.75", "r2": 0.78},
    ...
  },
  "ordinality_check": {
    "TV": {"monotonic": true, "spearman_rho": 0.85},
    ...
  },
  "multicollinearity": {
    "VIF": {"TV": 2.1, "Digital": 3.5, ...},
    "high_collinearity_pairs": [["TV", "Sponsorship"]],
    "recommended_combinations": [["TV", "Sponsorship"]]
  }
}
```

### Validation Rules
- ✅ All transformations increase r2 vs linear or maintain r2 >= 0.5
- ✅ Ordinality preserved for key channels
- ✅ VIF < threshold for most features (warn if > threshold)
- ❌ Fail if transformation produces NaN/Inf values

### Edge Cases
- **Constant channel** → Transformation undefined, flag as "no variance"
- **Single observation** → Cannot fit curve, return identity transform
- **Negative values in log transform** → Add small constant (log(x+1))
- **High collinearity** → Suggest combining; allow manual override

### Performance
- Curve fitting (per channel): ~100-500ms (depending on # of observations)
- Full feature engineering: <10 seconds for 50MB data

### Future Vision (V2)
- Automatic feature interaction detection (e.g., TV × Digital)
- LLM explanation: "We're using log(TV) because diminishing returns are evident: higher budgets yield less incremental revenue"
- Domain-informed transformations from user prompts
