# Agentic MMIX Pipeline — Low-Level Design & System Documentation

# Part 3: Feature Registry & Model Architecture

---

## 1. Raw Data Features

The pipeline ingests 7 CSV files containing the following key variables used in modeling:

### 1.1 Target Variable

| Feature | Source File | Type | Range (Monthly) | Description |
|---------|-----------|------|-----------------|-------------|
| total_gmv | SecondFile.csv / firstfile.csv | Continuous (₹) | 18 Cr – 51 Cr | Gross Merchandise Value — total revenue. Primary target for all MMIX models. |

### 1.2 Media Channel Spend (9 Channels)

| Channel | Source | Monthly Range (₹) | GMV Correlation | Group Assignment |
|---------|--------|-------------------|----------------|-----------------|
| TV | SecondFile.csv | Variable | +0.683 (strong) | traditional |
| Sponsorship | SecondFile.csv | Variable | +0.726 (strong) | traditional |
| Online.marketing | SecondFile.csv | Variable | +0.887 (strong) | digital_performance |
| Affiliates | SecondFile.csv | Variable | +0.866 (strong) | digital_performance |
| Digital | SecondFile.csv | Variable | +0.390 (moderate) | digital_brand |
| Content.Marketing | SecondFile.csv | Variable | +0.617 (strong) | digital_brand |
| SEM | SecondFile.csv | Variable | +0.535 (moderate) | digital_brand |
| Radio | SecondFile.csv | Variable | -0.946 (strong neg) | other |
| Other | SecondFile.csv | Variable | -0.959 (strong neg) | other |

### 1.3 Control & Contextual Variables

| Feature | Source | Type | Range | Description |
|---------|--------|------|-------|-------------|
| NPS | MonthlyNPSscore.csv | Continuous | 44–60 | Net Promoter Score. Correlation with GMV: -0.96. Used as seasonality proxy (A4), not customer satisfaction measure. |
| total_Discount | SecondFile.csv | Continuous (₹) | Variable | Total discount amount across all products |
| total_Mrp | SecondFile.csv | Continuous (₹) | Variable | Total MRP (list price) — denominator for discount ratio |
| Total.Investment | SecondFile.csv | Continuous (₹) | Variable | Sum of all 9 channel spends |
| sale_flag | SpecialSale.csv | Binary (0/1) | 0–1 | Whether any sale event occurred in this period |
| sale_days | SpecialSale.csv | Integer | 0–7 | Number of unique sale dates in the period |
| sale_intensity | SpecialSale.csv | Integer | 0–3 | Number of unique sale event names in the period |

---

## 2. Engineered Features

### 2.1 Log Transforms (12 variables)

All spend-related variables are log-transformed using `log(x + 1)` to enable elasticity interpretation and handle zero-spend periods cleanly.

| Raw Column | Log Column | Purpose |
|-----------|-----------|---------|
| total_gmv | **log_total_gmv** | Model target in log-log and log-linear specs |
| TV | log_TV | Channel elasticity |
| Digital | log_Digital | Channel elasticity |
| Sponsorship | log_Sponsorship | Channel elasticity |
| Content.Marketing | log_Content.Marketing | Channel elasticity |
| Online.marketing | log_Online.marketing | Channel elasticity |
| Affiliates | log_Affiliates | Channel elasticity |
| SEM | log_SEM | Channel elasticity |
| Radio | log_Radio | Channel elasticity |
| Other | log_Other | Channel elasticity |
| Total.Investment | log_Total_Investment | Aggregate spend elasticity |
| total_Discount | log_total_Discount | Discount elasticity |

**Why log(x+1)?** The +1 ensures log(0) doesn't produce -∞ for zero-spend periods. In a log-log model, coefficients are directly interpretable as elasticities: a 1% increase in spend produces a β% change in GMV.

### 2.2 Channel Group Spend (8 variables)

Individual channels are aggregated into 4 groups based on **EDA correlation analysis** (not business convention alone). Both raw and log-transformed versions are created.

| Group | Constituent Channels | Raw Column | Log Column |
|-------|---------------------|-----------|-----------|
| traditional | TV + Sponsorship | spend_traditional | log_spend_traditional |
| digital_performance | Online.marketing + Affiliates | spend_digital_performance | log_spend_digital_performance |
| digital_brand | Digital + Content.Marketing + SEM | spend_digital_brand | log_spend_digital_brand |
| other | Radio + Other | spend_other | log_spend_other |

### 2.3 Channel Group Validation (from EDA)

The grouping was validated through three levels of correlation analysis:

**Level 1 — Intra-Group Correlations (must be high):**

| Group | Channel Pair | Correlation | Assessment |
|-------|-------------|-------------|------------|
| digital_performance | Online.marketing ↔ Affiliates | **r = 0.989** | Must group — essentially the same signal |
| digital_brand | Digital ↔ Content.Marketing | r = 0.900 | Strong — same strategic category |
| digital_brand | Digital ↔ SEM | **r = 0.973** | Very strong — SEM moved here from digital_perf |
| digital_brand | Content.Marketing ↔ SEM | r = 0.958 | Strong — confirms grouping |
| traditional | TV ↔ Sponsorship | r = 0.550 | Moderate — distinct enough for separate specs but same strategic category |

**Level 2 — Inter-Group Correlations (must not be too high):**

| Group Pair | Group-Level Correlation | Assessment |
|-----------|----------------------|------------|
| traditional ↔ digital_performance | 0.513 | OK — well below 0.7 |
| traditional ↔ digital_brand | 0.746 | Borderline but acceptable — different strategies |
| digital_performance ↔ digital_brand | 0.386 | OK — distinct signals |
| traditional ↔ other | -0.169 | OK — uncorrelated |
| digital_performance ↔ other | 0.335 | OK |
| digital_brand ↔ other | -0.174 | OK |

**Level 3 — Group-GMV Correlations:**

| Group | GMV Correlation | Interpretation |
|-------|----------------|---------------|
| digital_performance | **+0.884** | Strongest revenue driver |
| traditional | +0.751 | Strong second |
| digital_brand | +0.507 | Moderate — but convergence shows negative marginal impact |
| other | +0.214 | Weak (Radio/Other individually negative) |

**SEM Reassignment Decision:** SEM was originally in `digital_performance`. EDA revealed SEM correlates with Digital (r=0.97) and Content.Marketing (r=0.96) far more strongly than with Online.marketing (r=0.44). SEM was moved to `digital_brand` to reflect actual co-movement patterns.

### 2.4 Promotional Features

| Feature | Calculation | Type | Interpretation |
|---------|-----------|------|---------------|
| sale_flag | Binary: any sale event in period | 0/1 | Promotional period indicator |
| sale_days | Count of unique sale dates per period | Integer (0-7) | Promotional duration |
| sale_intensity | Count of unique sale event names per period | Integer (0-3) | Promotional depth |
| discount_intensity | total_Discount / total_Mrp | Ratio (0.38–0.50) | Price reduction depth. Avoids endogeneity from absolute amounts. |
| discount_per_unit | total_Discount / total_Units | Ratio | Per-unit discount — alternative measure |

### 2.5 Control & Momentum Features

| Feature | Calculation | Purpose | Notes |
|---------|-----------|---------|-------|
| nps_standardized | (NPS − mean) / std | Seasonality proxy (A4) | NOT customer satisfaction. Negative coefficient expected. Removing drops R² significantly. |
| total_gmv_lag1 | total_gmv shifted 1 period | Momentum / carryover | Captures whether last period's GMV predicts this period's. Costs 1 observation. |
| Total.Investment_lag1 | Total.Investment shifted 1 period | Spend carryover | Available but not in current specs — reserved for future adstock modeling. |

### 2.6 Seasonality Features (Available, Not in Specs)

| Feature | Calculation | Purpose | Status |
|---------|-----------|---------|--------|
| is_festival_season | 1 if month is Oct/Nov/Dec | Festival quarter indicator | Created but NOT in any spec. Was tested as NPS replacement but NPS proved more effective for R². |
| month_sin | sin(2π × month / 12) | Cyclical month encoding | Created but NOT in any spec. Avoids Dec→Jan discontinuity. |
| month_cos | cos(2π × month / 12) | Cyclical month encoding | Created but NOT in any spec. Pair with month_sin for full cycle. |

**Why not in specs?** Testing showed that removing NPS and adding seasonality features produced lower R² than keeping NPS. NPS absorbs seasonal variance more effectively because it captures the intensity of seasonal effects (varies month to month), while is_festival_season is binary and month_sin/cos are fixed patterns. NPS is kept with clear documentation that it's a seasonality proxy (A4).

### 2.7 Feature Matrix Summary

| Granularity | Rows | Total Columns | Specs | Features per Spec |
|-------------|------|--------------|-------|------------------|
| Weekly | 47 | ~46 | 10 (6 base + 4 weekly) | 3–8 |
| Monthly | 10 | ~24 | 6 (base only) | 3 |

Note: The matrix contains ALL features (raw, log, groups, controls, seasonality) — far more columns than any single spec uses. Each spec selects a specific subset.

---

## 3. Model Specifications (10 Curated)

### 3.1 Design Principles

1. **Each spec tests a distinct hypothesis** — no two specs test the same question
2. **Base specs (A-F):** 3 features maximum, safe for monthly n=10. Tested in iteration 1 ("base" strategy).
3. **Weekly specs (G-J):** 6-8 features, group-based. Only added when n≥30. Tested in iteration 2 ("expanded" strategy, delta only).
4. **No individual channels in weekly specs:** EDA showed inter-channel correlations of r=0.97-0.99, making individual channel coefficients unreliable. Weekly specs use channel groups instead, with additional control features (discount, lag, NPS) to compensate for fewer channel dimensions.
5. **Channel selection is data-driven:** Spec C uses top-2 channels ranked by |correlation| with log_total_gmv. No hardcoded channel names.
6. **NPS included as seasonality proxy** where needed (specs B, F, G, H, J) per Assumption A4.

### 3.2 Base Specs (A-F) — Always Available

| Spec | Target | Features | Count | Hypothesis |
|------|--------|----------|-------|-----------|
| **A** | log_total_gmv | log_spend_traditional, log_spend_digital_performance, sale_flag | 3 | Do the two strongest channel groups (traditional + digital_perf) plus sale events explain GMV? |
| **B** | log_total_gmv | log_Total_Investment, sale_flag, nps_standardized | 3 | Does aggregate spend with sale and seasonality control explain GMV? This is the simplest meaningful baseline. |
| **C** | log_total_gmv | top2_channels (data-driven), sale_flag | 3 | Can just the best 2 individual channels (by |corr| with GMV) explain GMV without aggregation? Tests if granular channel data adds value over groups. |
| **D** | log_total_gmv | log_spend_digital_performance, sale_flag, total_gmv_lag1 | 3 | Does the strongest channel group with momentum (last period's GMV) predict better than without? Tests carryover effects. |
| **E** | log_total_gmv | log_Total_Investment, discount_intensity, sale_flag | 3 | Does promotional depth (discount ratio) drive revenue beyond what total spend and sale events capture? Tests the discounting lever. |
| **F** | log_total_gmv | log_Total_Investment, sale_days, nps_standardized | 3 | Does the number of sale days matter more than the binary sale flag? Tests whether promotional duration drives GMV beyond just having a sale. |

### 3.3 Weekly Specs (G-J) — Added When n ≥ 30

| Spec | Target | Features | Count | Hypothesis |
|------|--------|----------|-------|-----------|
| **G** | log_total_gmv | log_spend_traditional, log_spend_digital_performance, log_spend_digital_brand, sale_flag, discount_intensity, nps_standardized | 6 | Do the 3 main channel groups with promotional depth and seasonality control explain GMV better than aggregate spend? Tests group-level attribution. |
| **H** | log_total_gmv | log_spend_traditional, log_spend_digital_performance, log_spend_digital_brand, sale_flag, total_gmv_lag1, nps_standardized | 6 | Same as G but with momentum instead of discount. Tests whether carryover effects improve group-level models. |
| **I** | log_total_gmv | log_spend_traditional, log_spend_digital_performance, log_spend_digital_brand, log_spend_other, sale_flag, discount_intensity, total_gmv_lag1 | 7 | All 4 channel groups with both controls (discount + lag). Tests if including the "other" group (Radio/Other, negative correlation) improves or hurts the model. |
| **J** | log_total_gmv | log_spend_traditional, log_spend_digital_performance, log_spend_digital_brand, log_spend_other, sale_flag, discount_intensity, nps_standardized, total_gmv_lag1 | 8 | Maximum information: all 4 groups + all 3 controls. The most complex spec — relies on Ridge/Lasso/ElasticNet to handle multicollinearity and select relevant features. |

### 3.4 Spec-Feature Matrix

| Feature | A | B | C | D | E | F | G | H | I | J |
|---------|---|---|---|---|---|---|---|---|---|---|
| log_spend_traditional | ✓ | | | | | | ✓ | ✓ | ✓ | ✓ |
| log_spend_digital_performance | ✓ | | | ✓ | | | ✓ | ✓ | ✓ | ✓ |
| log_spend_digital_brand | | | | | | | ✓ | ✓ | ✓ | ✓ |
| log_spend_other | | | | | | | | | ✓ | ✓ |
| log_Total_Investment | | ✓ | | | ✓ | ✓ | | | | |
| top2 channels (dynamic) | | | ✓ | | | | | | | |
| sale_flag | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ |
| sale_days | | | | | | ✓ | | | | |
| discount_intensity | | | | | ✓ | | ✓ | | ✓ | ✓ |
| nps_standardized | | ✓ | | | | ✓ | ✓ | ✓ | | ✓ |
| total_gmv_lag1 | | | | ✓ | | | | ✓ | ✓ | ✓ |
| **Total features** | **3** | **3** | **3** | **3** | **3** | **3** | **6** | **6** | **7** | **8** |

### 3.5 Why No Individual Channels in Weekly Specs

EDA correlation analysis revealed extreme inter-channel correlations that make individual channel coefficients unreliable in multi-channel regressions:

| Channel Pair | Correlation | Problem |
|-------------|-------------|---------|
| Online.marketing ↔ Affiliates | **0.989** | Essentially the same signal — OLS cannot separate |
| SEM ↔ Digital | 0.973 | Near-identical spending patterns |
| Content.Marketing ↔ SEM | 0.958 | Co-movement too strong |
| Digital ↔ Content.Marketing | 0.900 | High overlap |

With correlations above 0.9, individual channel coefficients become highly unstable — small data perturbations can flip signs and magnitudes. VIF values for specs using multiple individual channels exceeded 1000.

**Solution:** Weekly specs use channel GROUPS instead. Groups aggregate the correlated channels, producing VIF values under 10 for most specs. Ridge/Lasso provide additional regularization for remaining multicollinearity.

Spec C is the exception — it uses 2 individual channels (data-driven top-2 by |correlation|). This works because only 2 channels are included simultaneously, keeping VIF manageable.

---

## 4. Model Architecture

### 4.1 Algorithm Types (6 Linear Models)

| Model | Library | Type | Key Hyperparameters | Strengths | Weaknesses | Why Included |
|-------|---------|------|-------------------|-----------|------------|-------------|
| OLS | statsmodels | Linear | None | P-values, AIC/BIC, full interpretability | No regularization, unstable with multicollinearity | Baseline: standard MMIX approach, statistical significance |
| Ridge | scikit-learn | Linear | alpha=1.0 | L2 regularization, stable coefficients even with correlated features | No feature selection, no p-values | Handles group multicollinearity; typically the best performer |
| Lasso | scikit-learn | Linear | alpha=CV-tuned, max_iter=10000 | L1 regularization, automatic feature selection (zeros out weak features) | Can zero out important features if correlated with stronger ones | Natural feature selection for high-feature specs (G-J) |
| ElasticNet | scikit-learn | Linear | alpha=CV, l1_ratio=[0.1-0.9] | Combined L1+L2, balanced regularization | More hyperparameters to tune | Best of both worlds for mixed feature types |
| Bayesian Ridge | scikit-learn | Linear | Auto-tuned alpha/lambda | Uncertainty estimates, automatic regularization | Slower, no p-values | Confidence intervals on elasticities |
| Huber | scikit-learn | Linear | epsilon=1.35 | Robust to outliers — downweights high-residual points | No p-values, no feature selection | Sale-period GMV spikes are leverage points; Huber reduces their influence |

### 4.2 Why No Tree-Based Models

XGBoost and Random Forest were evaluated but **excluded from the final pipeline** because:

1. **No interpretable elasticities:** Tree models produce feature importance (relative ranking) but not coefficients (directional magnitude). MMIX requires knowing "a 10% increase in Sponsorship spend produces X% GMV lift" — trees cannot provide this.

2. **No coefficient signs:** Trees cannot tell you whether a channel has positive or negative marginal impact. The convergence-based ordinality check requires signed coefficients.

3. **Scenario simulation incompatibility:** The scenario simulator multiplies baseline features by channel change factors and re-predicts. This works naturally with linear models but produces unreliable results with trees (extrapolation outside training data range).

4. **LOO CV overhead:** With n=47 and tree models, LOO CV is slow (47 × full model fit). Linear models are fast.

The builder functions (`build_xgboost`, `build_random_forest`) remain in the codebase for potential future use with larger datasets where non-linear effects become detectable.

### 4.3 Transform Variants (2)

| Variant | Target | Features | Coefficient Interpretation | When Better |
|---------|--------|----------|--------------------------|------------|
| **log_log** | log(GMV+1) | log(spend+1) | **Elasticity:** 1% increase in spend → β% increase in GMV | Standard MMIX interpretation. Handles diminishing returns naturally. |
| **log_linear** | log(GMV+1) | raw spend | **Semi-elasticity:** 1 unit (₹) increase in spend → β% increase in GMV | When absolute spend amounts are more interpretable than percentages. |

The `LOG_TO_RAW_MAP` in config.py maps log feature names back to raw column names, enabling the modeling pipeline to resolve features correctly for each transform variant.

### 4.4 Total Model Candidates

| Iteration | Strategy | Granularity | Specs | Types | Transforms | Candidates |
|-----------|----------|-------------|-------|-------|------------|-----------|
| 1 | base | weekly | 6 (A-F) | 6 | 2 | **72** |
| 2 | expanded (delta) | weekly | 4 (G-J only) | 6 | 2 | **48** |
| 3 | monthly (fallback) | monthly | 6 (A-F) | 6 | 2 | **72** |
| **Max total** | | | | | | **192** |

Not all candidates succeed — some are skipped due to missing columns, insufficient rows, or build failures. Typical success rates: ~83% for weekly (60/72), ~50% for monthly (36/72 due to n=10 limiting complexity).

---

## 5. Evaluation Metrics

### 5.1 Composite Score Formula

```
Composite = 0.35 × Fit + 0.25 × Stability + 0.25 × Ordinality + 0.15 × VIF
```

| Component | Weight | Metric | Calculation | Why This Weight |
|-----------|--------|--------|-------------|----------------|
| **Fit** | 35% | Adjusted R² | `max(0, min(1, adj_r_squared))` | Primary: does the model explain GMV? Adjusted penalizes overfitting from extra predictors. |
| **Stability** | 25% | 1 − |Train R² − CV R²| / Train R² | `max(0, 1 - abs(tr2 - cv2) / tr2)` using LOO CV | Does the model generalize? With n=47, overfitting is a real risk. |
| **Ordinality** | 25% | Convergence-based (see below) | 1.0 if all convergence-confirmed features match; penalized per violation | Business logic: coefficients should align with cross-model consensus. |
| **VIF** | 15% | Scaled by max VIF | ≤5: 1.0, ≤10: 0.7, ≤50: 0.3, >50: 0.1 | Multicollinearity reliability. Lower weight because Ridge/Lasso inherently handle it. |

### 5.2 Why These Weights

| Metric | Why Chosen | Business Relevance |
|--------|-----------|-------------------|
| Adjusted R² | Penalizes overfitting from extra predictors | With n=47, adding features always improves R² but may not generalize |
| LOO CV | Leave-one-out is the most thorough CV for small samples | Each data point gets a turn as test set — no lucky/unlucky splits |
| Convergence ordinality | Uses cross-model consensus instead of hardcoded rules | Digital_brand is genuinely negative; hardcoded "must be positive" would penalize correct models |
| VIF | Multicollinearity detection | Lower weight because regularized models handle it; still useful for flagging extreme cases |

### 5.3 Convergence-Based Ordinality (Two-Pass System)

Traditional MMIX enforces "all spend coefficients must be ≥ 0." This fails for:
- Channels with negative marginal impact when controlling for others (digital_brand)
- Channels with negative raw correlation (Radio, Other)
- NPS (seasonality proxy with expected negative coefficient)

**The two-pass system uses cross-model consensus as the truth:**

**Pass 1 — Build and Rank All Models:**
Build all (spec × type × transform) candidates. Initial ordinality check only enforces positive on unambiguous features: `sale_flag`, `sale_days`, `Total.Investment`. All channel/group features have no constraint. Rank by composite score.

**Pass 2 — Convergence Re-Scoring:**
1. Compute convergence across ALL successful models:
   - For each feature: collect coefficients from all models containing it
   - If ≥ 80% of models have positive coefficient → "POSITIVE (confirmed)"
   - If ≤ 20% positive → "NEGATIVE (confirmed)"
   - Otherwise → "MIXED (inconclusive)"

2. Re-score ordinality for EACH model:
   - For each confirmed feature in the model: does the coefficient match the confirmed direction?
   - Match → PASS for that feature
   - Mismatch → VIOLATION
   - Ordinality score = proportion of checks that pass

3. Re-compute composite score with updated ordinality. Re-rank all models.

**Example:**

| Feature | Convergence Direction | Model A Coef | Model A Check | Model B Coef | Model B Check |
|---------|---------------------|-------------|--------------|-------------|--------------|
| sale_flag | POSITIVE (confirmed) | +0.22 | ✅ PASS | +0.18 | ✅ PASS |
| log_spend_traditional | POSITIVE (confirmed) | +0.17 | ✅ PASS | -0.05 | ❌ FAIL |
| log_spend_digital_brand | NEGATIVE (confirmed) | -0.15 | ✅ PASS | +0.03 | ❌ FAIL |
| nps_standardized | NEGATIVE (confirmed) | -0.04 | ✅ PASS | +0.01 | ❌ FAIL |

Model A: 4/4 pass → ordinality score = 1.0
Model B: 1/4 pass → ordinality score = 0.25

This ensures Model A (which correctly captures digital_brand as negative) ranks higher than Model B (which has the "intuitive" but incorrect positive digital_brand coefficient).

---

## 6. Cross-Validation Strategy

### 6.1 Leave-One-Out (LOO) CV

**Why LOO?** With n=47 (weekly) or n=10 (monthly), k-fold CV has high variance — which 5 or 10 observations end up in the test fold can dramatically change results. LOO eliminates this randomness: every observation gets exactly one turn as the test point.

**Process:**
For each of n observations:
1. Remove observation i from training data
2. Fit model on remaining n-1 points
3. Predict observation i
4. Record prediction

After all n rounds:
- CV R² = R² of actual vs all held-out predictions
- CV RMSE = RMSE of held-out predictions
- CV MAPE = Mean Absolute Percentage Error of held-out predictions

**Cost:** n model fits per candidate. For 72 candidates × 47 LOO folds = 3,384 model fits per iteration. Linear models make this feasible (~2-3 minutes total).

### 6.2 Stability Score

```
Stability = max(0, 1 - |Train_R² - CV_R²| / Train_R²)
```

A model with Train R² = 0.58 and CV R² = 0.36 has stability = 1 - |0.58-0.36|/0.58 = 0.62. This is moderate — the model explains less on held-out data than training data, indicating some overfitting.

Perfect stability (1.0) means CV R² equals Train R². Zero stability means CV R² is zero or negative.

---

## 7. Scenario Simulation Architecture

### 7.1 Simulator Design

The scenario simulator translates raw business decisions ("increase Sponsorship by 20%") into model-level feature changes, predicts the outcome, and converts back to business metrics.

**Translation chain:**
```
Raw change: {"Sponsorship": 1.2}
    ↓ (if log-log transform)
Feature change: log(Sponsorship × 1.2 + 1) - log(Sponsorship + 1)
    ↓ (if model uses groups)
Group update: spend_traditional = TV + (Sponsorship × 1.2)
    ↓
log_spend_traditional = log(new_spend_traditional + 1)
    ↓
Model prediction → log(GMV_predicted)
    ↓
GMV_predicted = exp(log_pred) - 1 (if log target)
    ↓
Change % = (GMV_predicted - GMV_baseline) / GMV_baseline × 100
```

### 7.2 Standard Scenarios (11+)

| Category | Scenario | Changes Applied |
|----------|---------|----------------|
| **Individual channels** | +20% Online.marketing | Online.marketing × 1.2 |
| | +20% Sponsorship | Sponsorship × 1.2 |
| | +20% TV | TV × 1.2 |
| | +20% SEM | SEM × 1.2 |
| | +20% Affiliates | Affiliates × 1.2 |
| | +20% Digital | Digital × 1.2 |
| **Budget reallocation** | Shift 10% TV → Online.mkt | TV × 0.9, Online.mkt × 1.1 |
| **Scale** | All channels +10% | All × 1.1 |
| | All channels +20% | All × 1.2 |
| | All channels -20% | All × 0.8 |
| **Promotional** | Sale event activation | sale_flag = 1, sale_days = 3 |
| | Sale + 10% all | sale_flag = 1 + all × 1.1 |

### 7.3 When Scenarios Run

Scenarios are **only run after the agentic loop approves the model** (Phase 8, post-loop). During the iterative loop (Phase 6), `skip_scenarios=True` avoids wasting compute on models that may be rejected.

This means scenarios always reflect the best available model, and the scenario plot (`scenarios.png`) shows results from the final approved model, not an intermediate iteration.
