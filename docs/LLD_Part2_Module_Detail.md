# Agentic MMIX Pipeline — Low-Level Design & System Documentation

# Part 2: Module-Level Detail (Function Registry)

---

## 1. config.py — Functions & Exports

### get_paths(data_dir=None, output_dir=None)

**Input:** Optional path overrides (str or None)

**Logic:** Resolves project root from script location (`os.path.dirname` two levels up from `src/config.py`). Constructs all paths relative to root. Accepts overrides for data and output directories.

**Output:**
```python
{
    "root": "/path/to/Hackathon",
    "data_dir": "/path/to/Hackathon/data",
    "output_dir": "/path/to/Hackathon/outputs",
    "plots_dir": "/path/to/Hackathon/outputs/plots",
    "reports_dir": "/path/to/Hackathon/outputs/reports"
}
```

### find_col(df, candidates)

**Input:** `df`: pd.DataFrame, `candidates`: List[str] e.g. `["gmv_new", "GMV", "gmv"]`

**Logic:** Iterates candidates list, returns first that exists in `df.columns`. Handles column name inconsistencies across datasets.

**Output:** str column name or None

### get_channel_cols(df)

**Input:** `df`: pd.DataFrame

**Logic:** Filters `MEDIA_CHANNELS` list against `df.columns`. Returns only channels present in the DataFrame.

**Output:** List[str] e.g. `["TV", "Digital", "Sponsorship", ...]`

### PipelineSummary class

Collects structured summaries from every pipeline step. The `get_full_summary()` output is the primary input for LLM narrative generation.

**Methods:**

| Method | Input | Output |
|--------|-------|--------|
| `add_step(step_name, summary, log=None, decisions=None)` | Step name + summary string | Stores in internal dict |
| `add_error(step_name, error_msg)` | Step name + error | Appends to errors list |
| `add_warning(step_name, warning_msg)` | Step name + warning | Appends to warnings list |
| `get_full_summary()` | — | Concatenated markdown string for LLM |
| `get_step_summary(step_name)` | Step name | Summary string for that step |

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `DATA_FILES` | Dict of 7 entries | File registry: `{"transactions": {"filename": "firstfile.csv", "date_cols": ["Date"], "sep": ","}, ...}` |
| `MEDIA_CHANNELS` | 9 strings | `["TV", "Digital", "Sponsorship", "Content.Marketing", "Online.marketing", "Affiliates", "SEM", "Radio", "Other"]` |
| `CHANNEL_GROUPS` | 4 groups | `{"traditional": ["TV", "Sponsorship"], "digital_performance": ["Online.marketing", "Affiliates"], "digital_brand": ["Digital", "Content.Marketing", "SEM"], "other": ["Radio", "Other"]}` |
| `LOG_TO_RAW_MAP` | 16 mappings | `{"log_TV": "TV", "log_Digital": "Digital", ..., "log_spend_traditional": "spend_traditional", ...}` |
| `MODEL_SETTINGS` | Dict | `{"max_predictors_monthly": 3, "max_predictors_weekly": 6, "vif_threshold": 10, "score_weights": {...}, "quality_thresholds": {"min_r2": 0.50, "min_adj_r2": 0.45, "max_vif": 50, "min_models_passed": 5, "min_ordinality_rate": 0.5}}` |
| `LLM_CONFIG` | Dict | All values read from environment variables: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_OPENAI_API_KEY` |

---

## 2. eda_pipeline.py — Functions

### load_all_data(data_dir=None)

**Input:** `data_dir`: str path to data folder

**Logic:** Reads 7 CSV files using `config.DATA_FILES` registry. Each file has its own separator, date columns, and format. Applies post-load fixes:
- Sales.csv: GMV column coerced from object to numeric
- Investment: Column names stripped of whitespace
- Products: Frequency column comma removal for numeric parsing

**Output:**
```python
{
    "transactions": DataFrame (1,578,079 × 9),
    "sales": DataFrame (1,048,575 × 13),
    "monthly": DataFrame (12 × 39),
    "special_sales": DataFrame (44 × 2),
    "investment": DataFrame (12 × 12),
    "nps": DataFrame (12 × 2),
    "products": DataFrame (75 × 3)
}
```

### run_full_eda(data_dir=None, save_dir=None, data=None)

**Input:** `data_dir`: path, `save_dir`: plot output path, `data`: optional pre-loaded data dict

**Logic:** If `data` is provided, skips `load_all_data()` — eliminates the double-load problem. Runs 11 analysis steps sequentially, each wrapped in try/except to allow pipeline to continue if individual analyses fail. Steps:

1. Load data (or use pre-loaded)
2. Auto-classify columns
3. Validate data (missing columns, duplicates, negatives, anomalies)
4. Data quality report per dataset
5. National trends analysis + plot
6. Category breakdown + plot
7. Media investment analysis + plot
8. Correlation analysis + **correlation intelligence** + plot
9. Special sale impact + plot
10. Channel overlap analysis + plot
11. NPS vs revenue analysis + plot

**Output:** `(data, classifications, issues, corr_matrix)`

### auto_classify_columns(df, dataset_name="")

**Input:** `df`: pd.DataFrame, `dataset_name`: str

**Logic:** Classifies each column by keyword matching against known patterns:
- gmv, revenue → "revenue"
- unit, quantity → "units"
- nps, score → "score"
- tv, digital, sponsorship, content, online, affiliate, sem, radio → "investment"
- date, month, year → "date"
- discount, mrp, price → "pricing"

**Output:** Dict of `{column_name: {"type": str, "confidence": str}}`

### validate_mmix_data(data)

**Input:** `data`: Dict[str, DataFrame]

**Logic:** Runs validation checks across all datasets:
- Required columns present (revenue, units, investment, date, score)
- Duplicate dates check
- Negative values check per numeric column
- Sample size warning (n < 24 months)
- Channel nulls check
- Anomaly detection (GMV < 5% of median → flags dates)

**Output:** List of issue strings

### correlation_analysis(monthly_df, save_dir=None)

**Input:** `monthly_df`: pd.DataFrame, `save_dir`: path

**Logic:** Computes full correlation matrix for target + 9 channels + extras (Discount, NPS, Total.Investment). Generates heatmap and bar chart. **New correlation intelligence:**

1. **Feature-GMV sign classification:** Categorizes each channel as positive (corr > 0.1), negative (corr < -0.1), or weak/ambiguous
2. **Inter-channel correlations:** Flags all channel pairs with |r| > 0.85 as grouping candidates. Prints sorted list.
3. **Channel GROUP correlations:** Computes spend for each group from `CHANNEL_GROUPS`, builds group-level correlation matrix, flags pairs with |r| > 0.7. Validates that groups are distinct enough for regression.
4. **NPS context:** Reports NPS-GMV correlation and NPS-channel correlations. Flags if NPS-GMV < -0.5 as seasonality artifact.

All printed to console for immediate visibility. Correlation matrix returned and stored on `state.corr_matrix` for dynamic extraction by LLM evaluation prompt.

**Output:** Correlation matrix DataFrame

---

## 3. outlier_detection.py — Functions

### ASSUMPTIONS (module-level constant)

Dict of 9 documented assumptions (A1-A9). Each entry:
```python
{
    "A1_AUG2015": {
        "description": "Aug 2015 shows extremely low GMV...",
        "decision": "EXCLUDE from modeling...",
        "reasoning": "The platform was new..."
    },
    ...
}
```

### clean_monthly(monthly_df)

**Input:** `monthly_df`: pd.DataFrame (12 rows)

**Logic:** Sequential cleaning steps:
1. Sort by date, remove duplicate dates
2. Exclude Aug 2015 if GMV < 5% of median (Assumption A1)
3. Set negative values to 0 in all numeric columns
4. Coerce all channel columns to numeric, fill NaN with 0 (Assumption A3)
5. Validate NPS range [-100, 100]
6. Cap discount at MRP (discount > MRP → set to MRP)

**Output:** `(cleaned_df: pd.DataFrame, log: List[Dict], validation: Dict)` where validation has keys: `duplicate_dates`, `aug_2015_excluded`, `channel_zeros`, `nps_range`, `discount_valid`

### clean_weekly(weekly_df)

**Input:** `weekly_df`: pd.DataFrame (~48 rows from aggregation)

**Logic:**
1. Remove duplicate weeks
2. Remove partial first/last weeks (< 30% of median transactions)
3. Remove zero-GMV weeks
4. Remove anomalous weeks (< 5% of median GMV)
5. Set negatives to 0
6. Coerce channels to numeric
7. Validate NPS

**Output:** `(cleaned_df: pd.DataFrame, log: List[Dict])`

### detect_outliers_iqr(df, columns=None, multiplier=1.5)

**Input:** `df`: pd.DataFrame, `columns`: List[str], `multiplier`: float

**Logic:** Per column: compute Q1, Q3, IQR. Bounds = Q1 − 1.5×IQR to Q3 + 1.5×IQR. Flag values outside bounds.

**Output:** `(flags_df: pd.DataFrame, summary: Dict)` where flags_df has `{col}_outlier` binary columns

### detect_outliers_zscore(df, columns=None, threshold=2.5)

**Input:** `df`: pd.DataFrame, `columns`: List[str], `threshold`: float

**Logic:** Per column: z = |x − mean| / std, flag where z > threshold.

**Output:** `(flags_df: pd.DataFrame, summary: Dict)`

### business_context_review(df, special_sales_df=None)

**Input:** `df`: pd.DataFrame with Date and total_gmv, `special_sales_df`: pd.DataFrame

**Logic:** Per row: compute GMV/median ratio.
- If < 10% → EXCLUDE
- If > 200% and sale month → KEEP (sale-driven spike)
- If > 200% and no sale → FLAG for manual review

**Output:** List[Dict] with keys: `month`, `ratio`, `is_sale`, `action`, `reason`

### reconciliation_checks(monthly_df, investment_df=None, tolerance=0.05)

**Input:** `monthly_df`: pd.DataFrame, `investment_df`: pd.DataFrame

**Logic:** Cross-validates:
- sum(Revenue_*) ≈ total_gmv (revenue rollup)
- sum(Units_*) ≈ total_Units (units rollup)
- sum(channels) ≈ Total.Investment (investment rollup)
- Cross-validate channel totals with MediaInvestment.csv

**Output:** Dict with keys: `revenue_rollup`, `units_rollup`, `investment_rollup`, `cross_dataset` (all % difference)

### run_outlier_pipeline(data, granularity="weekly", save_dir=None)

**Input:** `data`: Dict of DataFrames, `granularity`: str, `save_dir`: path

**Logic:** Master function:
1. Print all assumptions
2. Clean monthly data
3. Run IQR + Z-score detection on key columns (total_gmv, Total.Investment, NPS)
4. Business context review
5. Reconciliation checks
6. Generate outlier summary plot
7. Return cleaned data with log of all actions

**Output:** `(clean_data: Dict, log: List[Dict], ASSUMPTIONS: Dict)`

---

## 4. data_aggregation.py — Functions

### build_modeling_dataset(clean_data, granularity="weekly")

**Input:** `clean_data`: Dict[str, DataFrame], `granularity`: str ("weekly" or "monthly")

**Logic:** Routes to `_build_weekly()` or `_build_monthly()` based on granularity. Returns standardized output dict.

**Output:** `{"data": Dict[str, DataFrame], "n_periods": int, "summary": str, "log": List}`

### _build_weekly(clean_data)

**Input:** `clean_data`: Dict with "transactions", "monthly", "special_sales", "nps"

**Logic:** Multi-step aggregation:
1. Parse transaction dates, create `week_start` (Monday of each week)
2. Group transactions by week_start: sum GMV, units, discount, MRP; count transactions
3. Distribute monthly channel spend evenly across weeks within each month (divide by weeks-in-month)
4. Map 44 sale events to weeks via `_map_sale_events_weekly()`
5. Add monthly NPS to weekly via year-month join `_add_weekly_nps()`
6. Add derived columns: Total.Investment (sum of channels), Year, Month
7. Clean weekly data via `clean_weekly()` from outlier_detection.py

**Output:** Weekly DataFrame (~47 rows × 30+ columns)

### _map_sale_events_weekly(weekly_df, special_sales_df)

**Input:** `weekly_df`: pd.DataFrame, `special_sales_df`: pd.DataFrame (44 events)

**Logic:** Map each sale date to its `week_start`. Group by week:
- `sale_days` = count of unique sale dates in that week
- `sale_intensity` = count of unique sale event names in that week
- `sale_flag` = binary (any sale in this week)

**Output:** `weekly_df` with `sale_flag`, `sale_days`, `sale_intensity` columns

### _add_weekly_nps(weekly_df, monthly_df, nps_data=None)

**Input:** `weekly_df`: pd.DataFrame, `monthly_df`: pd.DataFrame

**Logic:** Create year-month period for both DataFrames. Left-join monthly NPS to weekly via period matching. Each week gets its month's NPS value.

**Output:** `weekly_df` with `NPS` column

### _add_derived_columns(weekly_df)

**Input:** `weekly_df`: pd.DataFrame

**Logic:** If `Total.Investment` missing: sum all channel columns. Add `Year`, `Month`, `month` (label) columns.

**Output:** `weekly_df` with `Total.Investment`, `Year`, `Month`, `month` columns

---

## 5. feature_engineering.py — Functions

### apply_log_transforms(df, target_col="total_gmv")

**Input:** `df`: pd.DataFrame, `target_col`: str

**Logic:** For target + each of 9 channels + Total.Investment + total_Discount: create `log(x+1)` column. The +1 handles zero-spend periods cleanly. Enables elasticity interpretation in log-log models.

**Output:** `{"data": df, "log": [...], "summary": str, "decisions": []}` — df gains 12 columns: `log_total_gmv`, `log_TV`, `log_Digital`, ..., `log_Total_Investment`, `log_total_Discount`

### create_sale_features(df, special_sales_df)

**Input:** `df`: pd.DataFrame, `special_sales_df`: pd.DataFrame or None

**Logic:** If sale columns already exist (from weekly aggregation): reuse them, coerce to int. Otherwise: group SpecialSale.csv by month, merge sale_days/sale_intensity/sale_flag.

**Output:** `{"data": df, ...}` — df gains: `sale_flag`, `sale_days`, `sale_intensity`

### create_channel_groups(df)

**Input:** `df`: pd.DataFrame

**Logic:** For each group in `CHANNEL_GROUPS`: sum constituent channels into `spend_{group}`. Then create `log_spend_{group}` = log(spend+1). Produces 4 raw group spend columns + 4 log group spend columns.

Channel grouping rationale (from EDA):
- digital_performance: Online.marketing + Affiliates (r=0.99, must group)
- digital_brand: Digital + Content.Marketing + SEM (r=0.90-0.97; SEM moved here based on correlation)
- traditional: TV + Sponsorship (r=0.55, distinct enough)
- other: Radio + Other (both negative GMV correlation)

**Output:** `{"data": df, ...}` — df gains 8 columns: `spend_traditional`, `spend_digital_performance`, `spend_digital_brand`, `spend_other`, `log_spend_traditional`, `log_spend_digital_performance`, `log_spend_digital_brand`, `log_spend_other`

### create_discount_features(df)

**Input:** `df`: pd.DataFrame

**Logic:** Creates `discount_intensity` = total_Discount / total_Mrp (ratio avoids endogeneity bias from absolute discount amounts). Also creates `discount_per_unit` = total_Discount / total_Units.

**Output:** `{"data": df, ...}` — df gains: `discount_intensity`, `discount_per_unit`

### create_lagged_features(df, lag_cols=None, n_lags=1)

**Input:** `df`: pd.DataFrame, `lag_cols`: List[str] (default: total_gmv, Total.Investment), `n_lags`: int

**Logic:** Sort by Date. For each column × each lag: `df[col].shift(lag)`. First `n_lags` rows will have NaN (dropped later in assembly).

**Output:** `{"data": df, ...}` — df gains: `total_gmv_lag1`, `Total.Investment_lag1`

### prepare_nps(df)

**Input:** `df`: pd.DataFrame

**Logic:** Standardize NPS as z-score: `(NPS − mean) / std`. Computes correlation with total_gmv for logging. NPS is kept as a seasonality proxy per Assumption A4 — coefficient interpreted as seasonal adjustment, not customer satisfaction impact.

**Output:** `{"data": df, ...}` — df gains: `nps_standardized`

### create_seasonality_features(df)

**Input:** `df`: pd.DataFrame

**Logic:** Creates three seasonality features:
- `is_festival_season`: 1 if month is Oct-Dec (Dussehra/Diwali/Christmas quarter)
- `month_sin`: sin(2π × month / 12) — cyclical encoding
- `month_cos`: cos(2π × month / 12) — cyclical encoding (avoids Dec→Jan discontinuity)

These features are available in the matrix but NOT included in any model spec currently. They were created as alternatives to NPS for seasonal control but NPS proved more effective (removing NPS drops R² more than adding seasonality features recovers).

**Output:** `{"data": df, ...}` — df gains: `is_festival_season`, `month_sin`, `month_cos`

### assemble_feature_matrix(df, drop_lags_na=True)

**Input:** `df`: pd.DataFrame, `drop_lags_na`: bool

**Logic:** The core function that builds model specifications. Steps:

1. **Data-driven channel ranking:** Compute |correlation| of each `log_{channel}` with `log_total_gmv`. Sort descending. This ranking feeds into spec C (top 2 channels). No hardcoded channel names.

2. **Define 10 model specs:** 6 base specs (A-F, 3 features each, monthly-safe) + 4 weekly specs (G-J, 6-8 features, group-based). Weekly specs are only added if `n_rows >= 30`. See Part 3 for full spec table.

3. **Validate specs:** For each spec, check that target and all features exist in the DataFrame. Skip specs with missing columns.

4. **Drop lag NaN rows:** If lag features exist, drop first `n_lags` rows (loses 1 observation for lag-1).

5. **Assemble matrix:** Keep all columns referenced by any spec, plus metadata (Date, Month), raw values, raw channels, and group spend columns.

**Output:**
```python
{
    "data": pd.DataFrame (47×46 weekly or 10×24 monthly),
    "feature_sets": {
        "spec_A_grouped_channels": {
            "target": "log_total_gmv",
            "features": ["log_spend_traditional", "log_spend_digital_performance", "sale_flag"],
            "description": "Hypothesis: Do channel groups explain GMV?"
        },
        ... # 10 total for weekly, 6 for monthly
    },
    "log": [...],
    "summary": str,
    "decisions": []
}
```

### check_multicollinearity(df, feature_cols)

**Input:** `df`: pd.DataFrame, `feature_cols`: List[str]

**Logic:** Calculate Variance Inflation Factor (VIF) for each feature using `statsmodels.stats.outliers_influence.variance_inflation_factor`. VIF > 10 = problematic multicollinearity.

**Output:** `{"vif": pd.DataFrame with columns [feature, VIF], "log": [...], "summary": str, "decisions": [...]}`

### run_feature_engineering(clean_data, save_dir=None)

**Input:** `clean_data`: Dict of DataFrames (from aggregation), `save_dir`: path

**Logic:** Master pipeline running 9 steps:
1. Log transforms (12 variables)
2. Sale features (flag, days, intensity)
3. Channel grouping (4 groups + log versions)
4. Discount features (intensity, per-unit)
5. Lagged variables (GMV lag-1, Investment lag-1)
6. NPS preparation (z-score standardization)
6b. Seasonality features (festival, cyclical — available but not in specs)
7. Assembly (10 specs, data-driven ranking, validation)
8. VIF check per spec
9. Visualizations (feature distributions, features vs target)

**Output:**
```python
{
    "data": matrix_DataFrame,
    "feature_sets": validated_specs_dict,
    "vif_results": {spec_name: vif_result_dict},
    "full_log": all_actions_list,
    "summaries": {step_name: summary_string},
    "decisions": all_decisions_list
}
```

---

## 6. modeling_engine.py — Functions

### get_model_types(model_filter="all")

**Input:** `model_filter`: str — "all", "linear", or comma-separated names

**Logic:** Returns filtered dict of 6 model builders. Tree-based models (XGBoost, Random Forest) are defined in code but excluded from the registry — they don't produce interpretable elasticities needed for marketing budget recommendations.

**Output:** Dict[str, Dict] with keys: `description`, `builder` (function reference)

```python
{
    "OLS":        {"description": "OLS -- baseline, p-values",    "builder": build_ols},
    "Ridge":      {"description": "Ridge -- L2, stable",          "builder": build_ridge},
    "Lasso":      {"description": "Lasso -- L1, feature selection","builder": build_lasso},
    "ElasticNet": {"description": "ElasticNet -- L1+L2",          "builder": build_elasticnet},
    "Bayesian":   {"description": "Bayesian Ridge -- uncertainty", "builder": build_bayesian},
    "Huber":      {"description": "Huber -- robust to outliers",  "builder": build_huber},
}
```

### build_ols(X, y, feature_names)

**Input:** `X`: np.ndarray (n, p), `y`: np.ndarray (n,), `feature_names`: List[str]

**Logic:** `sm.OLS` with constant added. Checks for NaN/Inf before fitting. Also fits standardized version (StandardScaler) to compute standardized coefficients for cross-feature comparison.

**Output:** Dict with: `model`, `scaler`, `coefficients`, `standardized_coefficients`, `pvalues`, `r_squared`, `adj_r_squared`, `aic`, `bic`, `predictions`, `residuals`, `success`

### build_ridge(X, y, feature_names)

**Input:** Same as OLS

**Logic:** StandardScaler → Ridge(alpha=1.0). L2 regularization stabilizes coefficients when features are correlated (key for channel group specs).

**Output:** Same structure as OLS (no p-values, no AIC/BIC)

### build_lasso(X, y, feature_names)

**Input:** Same as OLS

**Logic:** StandardScaler → LassoCV(cv=min(5, n-1), max_iter=10000). L1 regularization zeros out weak features — natural feature selection. Alpha tuned by cross-validation.

**Output:** Same structure + `alpha` (selected regularization strength)

### build_elasticnet(X, y, feature_names)

**Input:** Same as OLS

**Logic:** StandardScaler → ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]). Combined L1+L2. Both alpha and l1_ratio tuned by CV.

**Output:** Same structure + `alpha`, `l1_ratio`

### build_bayesian(X, y, feature_names)

**Input:** Same as OLS

**Logic:** StandardScaler → BayesianRidge(). Automatic regularization with uncertainty estimates. Alpha and lambda tuned during fitting.

**Output:** Same structure (no p-values)

### build_huber(X, y, feature_names)

**Input:** Same as OLS

**Logic:** StandardScaler → HuberRegressor(epsilon=1.35, max_iter=200). Robust regression that downweights data points with high residuals — useful for sale-period GMV spikes that might be leverage points.

**Output:** Same structure (no p-values)

### check_ordinality(coefficients, feature_names, feature_correlations=None)

**Input:** `coefficients`: Dict[str, float], `feature_names`: List[str], `feature_correlations`: Dict[str, float] (optional, for logging)

**Logic:** Only enforces positive sign on **unambiguous** features:
- `sale_flag`, `sale_days`, `sale_intensity`: MUST be ≥ 0 (more sales = more GMV, always)
- `log_total_investment`, `total.investment`: MUST be ≥ 0 (aggregate spend)

All channel groups, individual channels, NPS, discount, lag features: **no sign constraint**. These are logged but not enforced because:
- Multicollinearity can flip signs for positively-correlated channels
- Convergence shows some groups (digital_brand) are genuinely negative
- NPS is a seasonality proxy with expected negative coefficient

**Note:** This is the initial ordinality check. The two-pass system re-scores ordinality using convergence after all models are built (see `run_modeling_pipeline`).

**Output:** `{"passed": bool, "violations": List[str], "checks": List[Dict], "n_violations": int}`

### loo_cross_validation(X, y, type_name, builder)

**Input:** `X`, `y`: arrays, `type_name`: str, `builder`: function

**Logic:** Leave-One-Out cross-validation. For each fold: train on n-1 points, predict the held-out point. Computes CV R², CV RMSE, CV MAPE from all held-out predictions.

**Output:** `{"cv_r2": float, "cv_rmse": float, "cv_mape": float}`

### score_model(train_result, cv_result, ordinality, vif_max=None)

**Input:** Train result dict, CV result dict, ordinality dict, max VIF value

**Logic:** Composite score with configurable weights from `MODEL_SETTINGS`:
```
Composite = 0.35 × Fit + 0.25 × Stability + 0.25 × Ordinality + 0.15 × VIF

where:
  Fit = max(0, min(1, adj_r_squared))
  Stability = max(0, 1 - |train_r2 - cv_r2| / train_r2) if cv_r2 else 0
  Ordinality = 1.0 if passed, else (1 - n_violations / n_checks)
  VIF = 1.0 if max_vif ≤ 5, 0.7 if ≤ 10, 0.3 if ≤ 50, 0.1 if > 50
```

**Output:** `{"composite": float, "fit_score": float, "stability_score": float, "ordinality_score": float, "vif_score": float, "vif_max": float}`

### analyze_convergence(results)

**Input:** `results`: List of all model result dicts

**Logic:** For each feature across all successful models: collect all coefficients. Compute:
- Mean coefficient
- % of models where coefficient is positive
- Direction: "POSITIVE (confirmed)" if ≥ 80% positive, "NEGATIVE (confirmed)" if ≤ 20% positive, "MIXED (inconclusive)" otherwise

**Output:** `{"insights": {feature: {"mean_coefficient": float, "n_models": int, "pct_positive": float, "direction": str}}, "log": [...], "summary": str}`

### build_scenario_simulator(model_result, feature_matrix, feature_names, target_col, full_monthly_data=None)

**Input:** Best model result, feature matrix, feature names, target column, optional monthly data for raw channel means

**Logic:** Builds a simulator function that:
1. Takes raw channel multipliers (e.g., `{"TV": 1.2}` for +20%)
2. Translates channel names to whatever features the model actually uses (log transforms, group aggregations)
3. Updates the baseline feature vector
4. Predicts with the model
5. Converts back from log scale if needed (`np.expm1`)
6. Returns baseline GMV, predicted GMV, and % change

**Output:** Simulator function `simulate(changes)` → `{"scenario_name": str, "baseline_gmv": float, "predicted_gmv": float, "change_pct": float}`

### run_standard_scenarios(simulator)

**Input:** Simulator function from `build_scenario_simulator`

**Logic:** Runs 11 predefined business scenarios:
1. Baseline (no change)
2. +20% individual channels (Online.marketing, Sponsorship, TV, SEM, Affiliates, Digital)
3. +20% all channels
4. -10% all channels
5. Shift 10% TV → Online.marketing
6. Activate sale event (sale_flag = 1)
7. Sale + 10% all channels
8. +50% digital performance group

**Output:** List of scenario result dicts

### run_modeling_pipeline(fe_result, clean_data=None, save_dir=None, top_n_scenarios=1, model_filter="all", skip_scenarios=False)

**Input:** Feature engineering result, clean data dict, save directory, scenario options, model filter, skip_scenarios flag

**Logic:** Master pipeline:

**Step 1 — Build all models:**
For each (spec × model_type × transform): resolve features (log or raw based on transform), validate columns, build model, check initial ordinality, run LOO CV, compute VIF, score composite.

**Step 2 — Initial ranking:**
Sort all successful models by composite score descending.

**Step 3 — Convergence analysis:**
Analyze coefficient direction agreement across all models. Identify confirmed positive/negative features.

**Step 3b — Convergence-based ordinality re-scoring (Two-Pass):**
Using convergence as truth: for each model, check if its coefficients match the consensus direction for each confirmed feature. Models agreeing with consensus → full ordinality score. Models contradicting → penalized. Re-score composite and re-rank.

**Step 4 — Best model:**
Print details of rank-1 model: spec, type, transform, R², Adj R², CV R², MAPE, ordinality, coefficients.

**Step 5 — Scenarios (conditional):**
If `skip_scenarios=False`: build simulator for top model(s), run standard + interactive scenarios. If `skip_scenarios=True` (during agentic loop): skip entirely.

**Step 6 — Plots:**
Model rankings (top 10 bar + top 5 breakdown), best model diagnostics (actual vs pred, residuals, coefficients, time series). Scenario plot only if scenarios were run.

**Output:**
```python
{
    "ranked_models": List[Dict],      # All models sorted by composite
    "top_10": List[Dict],             # Top 10 for display
    "best_model": Dict,               # Rank-1 model with all details
    "convergence": Dict,              # Cross-model agreement analysis
    "scenarios": List[Dict],          # Scenario results (empty if skipped)
    "simulator": Function,            # Simulator for custom scenarios
    "all_scenario_results": Dict,     # Multi-model scenarios
    "all_simulators": Dict,           # Multi-model simulators
    "full_log": List[str],
    "summaries": Dict[str, str],
    "decisions": List[Dict]
}
```

---

## 7. narrative_generator.py — Functions

### get_llm_client()

**Input:** None (reads from `LLM_CONFIG`)

**Logic:** Creates AzureOpenAI client using endpoint, API key, and API version from config. Returns None if key is missing.

**Output:** AzureOpenAI client or None

### call_llm(client, prompt, system_prompt=None, max_tokens=1500, temperature=0.3)

**Input:** Client, user prompt, optional system prompt, token limit, temperature

**Logic:** Calls GPT-4.1 via Azure OpenAI chat completions API. Default system prompt: senior marketing analytics consultant with rules on business language, Crores currency, no fabricated numbers.

**Output:** Response text string

### NarrativeGenerator class

**Methods:**

| Method | Input | Context Passed | Output |
|--------|-------|---------------|--------|
| `narrate_eda(data_summary, corr_summary, sale_lift)` | Summary strings | — | EDA narrative |
| `narrate_outliers(cleaning_log, assumptions)` | Log + assumptions dict | — | Cleaning narrative |
| `narrate_features(summaries, vif_summary=, channel_ranking=)` | FE summaries + VIF + correlation data | Multicollinearity context, channel grouping rationale | Feature narrative |
| `narrate_modeling(best, top10, convergence, vif_info=, agent_context=)` | Model results + VIF + agent history | Agent iterations, quality progression, model selection note | Modeling narrative |
| `narrate_scenarios(scenario_results)` | Scenario list | — | Scenario narrative |
| `save_report(output_dir)` | Path | — | Saved markdown filepath |

### generate_all_narratives(pipeline_result, outlier_log=None, assumptions=None, corr_matrix=None, save=True, agent_context=None)

**Input:** Pipeline result dict, outlier log, assumptions, correlation matrix, save flag, agent context dict

**Logic:** Master convenience function. Creates NarrativeGenerator, calls all 5 narrate methods in sequence with full context:

1. **EDA narrative:** Passes data summary, channel correlations extracted from `corr_matrix`, sale lift percentage.

2. **Outlier narrative:** Passes cleaning log and assumptions dict.

3. **Feature narrative:** Extracts VIF summary from `fe_result["vif_results"]` (max VIF and high-count per spec). Extracts channel ranking from `corr_matrix` (channel-GMV correlations). Passes both to `narrate_features` so LLM can explain multicollinearity rationale.

4. **Modeling narrative:** Extracts VIF info for the best model's spec. Passes `agent_context` dict containing iterations, quality_scores, final_strategy, adjustments_made, decisions_summary. LLM includes agent iteration history in the narrative if multiple iterations occurred.

5. **Scenario narrative:** Passes scenario results list.

Saves all narratives as `mmix_narrative_report.md`.

**Output:** NarrativeGenerator instance with all narratives populated

---

## 8. agent_orchestrator.py — Functions

### AgentState class

**Constructor:** `AgentState(granularity="weekly", top_n_scenarios=1, model_filter="all")`

**Fields:**

```python
# Pipeline outputs
data: Dict[str, DataFrame]         # Raw loaded data (Phase 1)
clean_data: Dict[str, DataFrame]   # After outlier cleaning (Phase 3)
outlier_log: List[Dict]            # Cleaning action log
assumptions: Dict                   # A1-A9 assumptions
corr_matrix: pd.DataFrame          # Correlation matrix from EDA (Phase 2)
aggregated_data: Dict               # After aggregation (Phase 4)
fe_result: Dict                     # Feature engineering output (Phase 5)
model_result: Dict                  # Modeling output (Phase 6)
narrator: NarrativeGenerator        # Narrative generator (Phase 9)

# Agent tracking
current_phase: str                  # e.g. "modeling", "quality_evaluation"
iteration: int                      # 0 = pre-loop, 1+ = iteration number
max_iterations: int                 # Default 3
reasoning_trace: List[Dict]         # Full decision log (timestamped)
decisions: List[str]                # Flat list of decision strings
quality_scores: Dict[int, Dict]     # Per-iteration quality assessment

# Agentic strategy
spec_strategy: str                  # "base", "expanded", "groups", "monthly"
adjustments: Dict                   # Current iteration's adjustments
last_llm_verdict: Dict              # Most recent LLM evaluation response
```

**Methods:**

| Method | Input | Output |
|--------|-------|--------|
| `add_reasoning(phase, reasoning, decision, details=None)` | Phase name, reasoning text, decision string, optional details dict | Appends to reasoning_trace |
| `get_trace_summary()` | — | Human-readable trace string (printed at end, saved to file) |

### QualityEvaluator class

**Constructor:** `QualityEvaluator(llm_client=None)`

**HARD Thresholds** (sourced from `MODEL_SETTINGS["quality_thresholds"]` in config.py):

| Threshold | Config Key | Default | Effect |
|-----------|-----------|---------|--------|
| `MIN_R2` | `min_r2` | 0.50 | RETRY if R² below (weekly) |
| `MIN_ADJ_R2` | `min_adj_r2` | 0.45 | RETRY if Adj R² below |
| `MAX_VIF` | `max_vif` | 50 | Issue flagged |
| `MIN_MODELS_PASSED` | `min_models_passed` | 5 | RETRY if fewer models succeed |
| `MIN_ORDINALITY_RATE` | `min_ordinality_rate` | 0.5 | Issue flagged if weak convergence |

**Methods:**

| Method | Input | Output |
|--------|-------|--------|
| `evaluate_model_quality(model_result)` | Model result dict | `{"acceptable": bool, "score": float, "issues": List, "suggestions": List, "reasoning": str}` |
| `suggest_improvements(quality, state, llm_verdict=None)` | Quality dict + state + optional LLM verdict | Adjustments dict (also sets `state.spec_strategy`) |
| `llm_evaluate(model_result, state)` | Model result + state | `{"verdict": "ACCEPT"/"RETRY", "reasoning": str, "coefficient_assessment": str, "suggestions": List}` |

### Pipeline Tools

| Tool | Phase | Critical? | Failure Behavior |
|------|-------|-----------|-----------------|
| `tool_load_data(state)` | 1 | YES | Pipeline stops |
| `tool_run_eda(state)` | 2 | NO | Continue without EDA (uses pre-loaded data) |
| `tool_run_outliers(state)` | 3 | YES | Pipeline stops |
| `tool_run_aggregation(state)` | 4 | YES | Pipeline stops |
| `tool_run_features(state)` | 5 | YES | Pipeline stops. Filters specs by `state.spec_strategy` |
| `tool_run_modeling(state, skip_scenarios=True)` | 6 | YES | Pipeline stops. Scenarios skipped during loop |
| `tool_run_scenarios(state)` | 8 | NO | Continue without scenarios (post-loop) |
| `tool_run_narratives(state)` | 9 | NO | Continue without narratives (post-loop, passes agent_context) |

### evaluate_and_decide(state, evaluator)

**Input:** AgentState, QualityEvaluator

**Logic:**
1. Run rule-based evaluation → quality dict with score, issues, suggestions
2. Store quality in `state.quality_scores[iteration]`
3. Run LLM evaluation → verdict, reasoning, coefficient_assessment, suggestions
4. Store on `state.last_llm_verdict`
5. Decision logic:
   - Max iterations reached → PROCEED (accept best available)
   - LLM says RETRY → RETRY (LLM respected even if rules pass)
   - Rule-based not acceptable → RETRY
   - Both pass → PROCEED

**Output:** "PROCEED" or "RETRY"

### suggest_improvements(quality, state, llm_verdict=None)

**Input:** Quality assessment, AgentState, optional LLM verdict

**Logic:** Maps issues to concrete strategy changes:

| Issue | Current Strategy | Adjustment |
|-------|-----------------|------------|
| Low R² | base | → "expanded" (add group specs G-J) |
| Low R² | expanded | → "monthly" (switch granularity) |
| High VIF | any | → "groups" (A, J only) |
| Ordinality failure | any | strict_ordinality filter |
| Unstable (CV gap) | expanded | → "base" (revert to simpler) |
| LLM RETRY, no rule issues | base | → "expanded" (default escalation) |
| LLM RETRY, no rule issues | expanded | No further action possible |

Sets `state.spec_strategy` and `state.adjustments` directly.

**Output:** Adjustments dict

### run_agentic_pipeline(granularity, top_n_scenarios, model_filter, skip_eda, data_dir, output_dir)

**Input:** All pipeline configuration options

**Logic:** See Part 1, Section 5 for full execution flow.

**Output:** AgentState with all results and complete reasoning trace
