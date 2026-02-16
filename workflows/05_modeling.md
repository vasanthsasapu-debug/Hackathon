# Workflow: Model Selection & Ranking

## TLDR
Build incremental models (start with best channel, add others) → Run GLM, Bayesian, Fixed/Random Effects variants → Score on Fit (60%), Stability (20%), Simplicity (20%) → Rank top 10 → Return ranked models with narratives.

---

## Summary

### Objective
Identify best model(s) balancing fit, stability, and interpretability for budget optimization.

### Inputs
- Engineered DataFrame (from feature_engineering workflow)
- Model config: model_types (GLM, Bayesian, Fixed/Random Effects), cv_folds, ordinality_channels

### Engine Module
`src/mmix/engine/modeling.py` (consolidated)

### Steps
1. **Incremental Feature Selection** → Start with most important channel (highest correlation with Sales_Output), iteratively add others, track model improvement
2. **Model Training** → For each feature set, train:
   - GLM (Poisson, Gamma distributions)
   - Bayesian regression
   - Fixed Effects (by Entity_ID or Demographic_Segment)
   - Random Effects (mixed models)
3. **Cross-Validation** → k-fold (default k=5) to assess stability
4. **Ordinality Check** → For key channels, verify coefficients are monotonically ordered (e.g., SEM coef > Affiliates)
5. **Model Scoring** → Compute composite score: 60% R² (fit) + 20% CV_std (stability) + 20% Model_Complexity (simplicity)
6. **Ranking** → Sort by score, return top 10

### Outputs
```json
{
  "ranked_models": [
    {
      "rank": 1,
      "model_type": "GLM_Poisson",
      "features": ["TV", "Digital", "SEM"],
      "r2": 0.88,
      "cv_std": 0.05,
      "ordinality": true,
      "coefficients": {"TV": 0.12, "Digital": 0.15, ...},
      "score": 0.876
    },
    ...
  ],
  "best_model": {ranked_models[0]},
  "cross_validation_results": {...}
}
```

### Validation Rules
- ✅ All models have R² > 0.3 (minimum fit threshold)
- ✅ CV_std < 0.15 (model not overfitting)
- ✅ Ordinality satisfied for key channels
- ❌ Fail if no model meets minimum threshold (likely data/feature issue)
- ⚠️ Warn if top model's R² < 0.5 (model quality low; suggest re-engineering)

### Edge Cases
- **Collinear features cause singular matrix** → Drop most collinear feature, retrain
- **Ordinal constraint violated** → Relax for non-key channels, warn for key channels
- **Overfitting detected (CV_std high)** → Suggest feature reduction
- **All models perform equally** → Suggest simplest (fewest features)

### Performance
- Training 100 model variants: ~15-30 seconds
- Cross-validation: Linear in cv_folds

### Future Vision (V2)
- Automatic model selection based on ordinality violations (suggest feature removal or transformation)
- LLM critique: "Model rank 5 has better ordinality than rank 1; recommend switching"
- Bayesian model averaging (combine top 3-5 models for ensemble predictions)
