# Project Structure — Agentic MMIX Pipeline

> Detailed description of every file, module, and directory in the project.

---

## Directory Layout

```
Hackathon/
│
├── main.py                              # [ENTRY] CLI wrapper → calls run_agentic_pipeline()
├── requirements.txt                     # Python dependencies
├── README.md                            # Project overview and usage guide
├── .env                                 # API keys (gitignored)
├── .gitignore                           # Git ignore rules
│
├── src/                                 # ── Core Pipeline Modules ──
│   ├── config.py                        # Configuration hub (paths, channels, groups, LLM)
│   ├── eda_pipeline.py                  # Data loading + 11 EDA analyses + correlation intelligence
│   ├── outlier_detection.py             # Cleaning, detection, 9 assumptions (A1-A9)
│   ├── data_aggregation.py              # Weekly/monthly aggregation
│   ├── feature_engineering.py           # Transforms, 10 specs (data-driven), seasonality features
│   ├── modeling_engine.py               # 6 model types, convergence-based ordinality, scenarios
│   ├── narrative_generator.py           # GPT-4.1 narratives with VIF/correlation/agent context
│   └── agent_orchestrator.py            # Agentic pipeline: state machine, 3-layer evaluator, loop
│
├── data/                                # ── Input Data (gitignored) ──
│   ├── firstfile.csv                    # Transaction-level records
│   ├── Sales.csv                        # Sales data
│   ├── SecondFile.csv                   # Monthly aggregated metrics
│   ├── SpecialSale.csv                  # Sale event dates
│   ├── MediaInvestment.csv              # Channel-level media spend
│   ├── MonthlyNPSscore.csv             # Net Promoter Scores
│   └── ProductList.csv                  # Product catalog
│
├── outputs/                             # ── Generated Outputs ──
│   ├── plots/                           # 13 PNG visualizations
│   └── reports/                         # Narrative report + reasoning trace
│
└── docs/                                # ── Documentation ──
    ├── architecture_diagram.html        # Interactive architecture diagram
    ├── architecture_diagram.mermaid     # Mermaid source
    ├── project_structure.md             # This file
    └── LLD_Part_[1-4]*.pdf             # Low-Level Design documents
```

---

## Entry Point

### `main.py` — CLI Wrapper

A thin wrapper that routes all runs through `run_agentic_pipeline()`. There is no separate "standard" pipeline — every run is agentic. If the model passes quality checks on iteration 1, the loop exits immediately with zero overhead.

| Attribute | Detail |
|-----------|--------|
| **Key function** | Imports and calls `run_agentic_pipeline()` from `agent_orchestrator.py` |
| **Both mode** | `run_both_granularities()` — runs agentic pipeline for weekly and monthly independently, then compares |
| **CLI flags** | `-g weekly/monthly/both`, `--skip-eda`, `--skip-narratives`, `-t <top_models>`, `-m <model_filter>`, `--data-dir`, `--output-dir` |
| **Comparison** | `_compare_granularities()` — side-by-side of both AgentState results (R², iterations, strategy, coefficients, convergence agreement) |

---

## Core Modules (src/)

### `config.py` — Configuration Hub

Single source of truth. All other modules import from here.

| Component | Contents |
|-----------|---------|
| **Paths** | `get_project_root()`, `get_paths(data_dir, output_dir)` → root/data/output/plots/reports |
| **DATA_FILES** | Registry of 7 CSV files with filenames, date columns, separators, format overrides |
| **MEDIA_CHANNELS** | 9 channels: TV, Digital, Sponsorship, Content.Marketing, Online.marketing, Affiliates, SEM, Radio, Other |
| **CHANNEL_GROUPS** | 4 groups validated by EDA correlation analysis: |
| | traditional: TV, Sponsorship |
| | digital_performance: Online.marketing, Affiliates (r=0.99) |
| | digital_brand: Digital, Content.Marketing, SEM (r=0.90-0.97; SEM moved here from digital_perf based on EDA) |
| | other: Radio, Other (both negative GMV correlation) |
| **LOG_TO_RAW_MAP** | Maps log-transformed column names to raw names for scenario simulation |
| **MODEL_SETTINGS** | Max predictors, VIF threshold, composite score weights (35/25/25/15) |
| **LLM_CONFIG** | Azure OpenAI endpoint, API version, deployment (GPT-4.1), API key |
| **PipelineSummary** | Collector class for step summaries → consumed by LLM narrative generation |
| **Helpers** | `find_col()`, `get_channel_cols()` |

---

### `eda_pipeline.py` — Data Loading & Exploratory Analysis

Handles Phase 1 (loading) and Phase 2 (EDA). Accepts pre-loaded data to avoid double loading.

| Function | Purpose | Output |
|----------|---------|--------|
| `load_all_data(data_dir)` | Load 7 CSVs using config registry, apply post-load fixes | Dict of 7 DataFrames |
| `run_full_eda(data_dir, save_dir, data=)` | Master EDA — accepts pre-loaded data, runs 11 analyses | (data, classifications, issues, corr_matrix) |
| `auto_classify_columns(df, name)` | Classify columns by type (revenue, units, investment, etc.) | Classifications dict |
| `validate_mmix_data(data)` | Missing columns, duplicates, negatives, anomalies | List of issue strings |
| `correlation_analysis(monthly)` | Full correlation matrix + **new correlation intelligence:** | Correlation matrix |
| | • Feature-GMV sign classification (positive vs negative channels) | |
| | • Inter-channel correlations (|r|>0.85 = grouping candidates) | |
| | • Channel GROUP correlations (validates that groups aren't too correlated) | |
| | • NPS seasonality detection and channel-NPS correlations | |
| `analyze_nps_revenue(monthly)` | NPS vs GMV trend | Stats + plot |
| `analyze_channel_overlap(monthly)` | Multi-channel co-occurrence | Overlap stats + plot |
| `analyze_category_breakdown(data)` | Revenue/units by category | Category stats + plot |
| `special_sale_impact(transactions, sales)` | GMV lift: sale vs non-sale days | Lift % + plot |
| `national_trends(monthly)` | Monthly GMV trend | Trend stats + plot |
| `media_investment_analysis(monthly)` | Channel spend distribution | Investment stats + plot |

**Key design decision:** EDA correlation findings are stored on `state.corr_matrix` and dynamically fed to the LLM evaluation prompt as context — the LLM learns about NPS seasonality, negative channels, and group correlations from the data, not from hardcoded instructions.

**Generates:** 7 PNG plots

---

### `outlier_detection.py` — Outlier Detection & Data Cleaning

Handles Phase 3. Documents 9 assumptions (A1-A9).

| Function | Purpose | Output |
|----------|---------|--------|
| `clean_monthly(monthly_df)` | Sort, exclude Aug 2015 (A1), set negatives to 0, coerce channels, validate NPS, cap discounts | (cleaned_df, log, validation) |
| `clean_weekly(weekly_df)` | Remove duplicate/partial/zero-GMV/anomalous weeks | (cleaned_df, log) |
| `detect_outliers_iqr(df, cols, mult)` | IQR method (1.5× multiplier) | (flags_df, summary) |
| `detect_outliers_zscore(df, cols, thresh)` | Z-score (2.5σ threshold) | (flags_df, summary) |
| `business_context_review(df, sales_df)` | GMV/median ratio → EXCLUDE/KEEP/FLAG based on sale context | List of action dicts |
| `reconciliation_checks(monthly, invest)` | Cross-validate rollups across datasets | Dict of % differences |
| `run_outlier_pipeline(data, gran, save_dir)` | Master: clean + detect + review + reconcile | (clean_data, log, ASSUMPTIONS) |

**9 Documented Assumptions:**

| ID | Description | Decision |
|----|-------------|----------|
| A1 | Aug 2015 GMV < 5% median | EXCLUDE (platform ramp-up) |
| A2 | Discount spikes during sales | KEEP (normal e-commerce) |
| A3 | Radio/Other zero spend months | KEEP zeros (budget decisions) |
| A4 | NPS -0.96 corr with GMV | KEEP as seasonality proxy (not causal) |
| A5 | Oct/Sep high GMV (festivals) | KEEP (sale_flag captures effect) |
| A6 | Discount may be endogenous | Use as CONTROL, not primary lever |
| A7 | Weekly n=47, monthly n=10 | Limit predictors accordingly |
| A8 | Sales.csv GMV dtype issues | Coerce to numeric during load |
| A9 | Radio/Other negative correlation | Allow negative coefficients |

**Generates:** `outlier_summary.png`

---

### `data_aggregation.py` — Weekly/Monthly Aggregation

Handles Phase 4. Transforms cleaned data into modeling-ready time series.

| Function | Purpose | Output |
|----------|---------|--------|
| `build_modeling_dataset(clean_data, gran)` | Route to weekly or monthly builder | {"data": DataFrame, "n_periods": int} |
| `_build_weekly(clean_data)` | Transaction-level → 7-day rollups (47 periods) | Weekly DataFrame |
| `_build_monthly(clean_data)` | Direct from cleaned monthly (10 periods) | Monthly DataFrame |
| `_map_sale_events_weekly(weekly, sales)` | Map 44 sale dates to week: sale_flag, sale_days, sale_intensity | Enriched DataFrame |
| `_add_weekly_nps(weekly, monthly, nps)` | Join monthly NPS to weekly via year-month | DataFrame + NPS |
| `_add_derived_columns(weekly)` | Total.Investment if missing, Year, Month | Enriched DataFrame |

**Note:** The agent can trigger re-aggregation to monthly on iteration 3 as a granularity fallback.

---

### `feature_engineering.py` — Feature Creation & Spec Building

Handles Phase 5. Creates all features, builds 10 curated model specifications with data-driven channel ranking.

| Function | Purpose | Output |
|----------|---------|--------|
| `apply_log_transforms(df, target)` | log(x+1) on target + channels + investment + discount (12 vars) | {"data": df, "log": [...]} |
| `create_sale_features(df, sales_df)` | sale_flag, sale_days, sale_intensity | {"data": df, "log": [...]} |
| `create_channel_groups(df)` | 4 spend groups (traditional, digital_perf, digital_brand, other) + log versions | {"data": df, "log": [...]} |
| `create_discount_features(df)` | discount_intensity (Discount/MRP), discount_per_unit | {"data": df, "log": [...]} |
| `create_lagged_features(df, lag)` | total_gmv_lag1, Total.Investment_lag1 | {"data": df, "log": [...]} |
| `prepare_nps(df)` | nps_standardized (z-scored NPS) — kept as seasonality proxy (A4) | {"data": df, "log": [...]} |
| `create_seasonality_features(df)` | is_festival_season, month_sin, month_cos — available but not in specs | {"data": df, "log": [...]} |
| `assemble_feature_matrix(df)` | Data-driven channel ranking + 10 spec definitions + validation | {"data": matrix, "feature_sets": specs} |
| `check_multicollinearity(df, cols)` | VIF calculation per spec | {"vif": DataFrame} |
| `run_feature_engineering(data, save_dir)` | Master: all steps + plots | Full result dict |

**Data-Driven Channel Ranking:** Channels are ranked by |correlation| with `log_total_gmv`. This ranking feeds into specs C (top 2 channels). No hardcoded channel names in any spec.

**10 Model Specs:**

| Spec | Features | Count | Type | Hypothesis |
|------|----------|-------|------|-----------|
| A | 2 groups + sale | 3 | Base | Channel groups explain GMV? |
| B | Total_Investment + sale + NPS | 3 | Base | Aggregate spend baseline |
| C | Top 2 channels (dynamic) + sale | 3 | Base | Best channels suffice? |
| D | Digital_perf + sale + lag | 3 | Base | Carryover/momentum? |
| E | Total_Investment + discount + sale | 3 | Base | Discounting matters? |
| F | Total_Investment + sale_days + NPS | 3 | Base | Duration vs binary flag? |
| G | 3 groups + sale + discount + NPS | 6 | Weekly | Group-level + controls |
| H | 3 groups + sale + lag + NPS | 6 | Weekly | Group-level + momentum |
| I | 4 groups + sale + discount + lag | 7 | Weekly | All groups + controls |
| J | 4 groups + sale + discount + NPS + lag | 8 | Weekly | Maximum information |

**Key design decisions:**
- Weekly specs use channel GROUPS, not individual channels (EDA showed inter-channel r=0.97-0.99)
- NPS kept as seasonality proxy despite negative coefficient (removing drops R² significantly)
- Seasonality features (festival season, cyclical month) created but not in specs — available for future use

**Generates:** `feature_distributions.png`, `features_vs_target.png`

---

### `modeling_engine.py` — Model Training, Scoring & Scenarios

Handles Phase 6 (modeling) and Phase 8 (scenarios, post-loop).

| Function | Purpose | Output |
|----------|---------|--------|
| `get_model_types(filter)` | 6 linear model builders (no tree-based — no interpretable elasticities) | Dict of builders |
| `get_transform_variants()` | log-log (elasticity) and log-linear (semi-elasticity) | Dict of transforms |
| `build_ols/ridge/lasso/elasticnet/bayesian/huber(X, y, names)` | Model builders with standardization | Train result dict |
| `check_ordinality(coefs, names, corr)` | Only enforces positive on unambiguous features (sale, Total.Investment) | Ordinality dict |
| `loo_cross_validation(X, y, type, builder)` | Leave-one-out CV → CV R², MAPE | CV result dict |
| `score_model(train, cv, ordin, vif)` | Composite: 35% fit + 25% stability + 25% ordinality + 15% VIF | Score dict |
| `analyze_convergence(results)` | Cross-model agreement on coefficient signs/magnitudes | Convergence dict |
| `build_scenario_simulator(result, ...)` | Build simulator accepting raw channel names | Simulator function |
| `run_standard_scenarios(simulator)` | 11 predefined scenarios | List of results |
| `run_modeling_pipeline(fe_result, ..., skip_scenarios=)` | Master: build → rank → **convergence → re-rank** → scenarios (optional) | Full result dict |

**Convergence-Based Ordinality (Two-Pass):**

| Pass | What Happens |
|------|-------------|
| Pass 1 | Build all models → initial rank → compute convergence (what ALL models agree on) |
| Pass 2 | Re-score ordinality: model coef matches convergence direction → PASS; contradicts → FAIL → Re-rank |

This means a model with negative digital_brand coefficient gets full ordinality score (convergence confirms it's negative), while a model with positive digital_brand contradicts consensus and gets penalized.

**`skip_scenarios` flag:** During the agentic loop, scenarios are skipped to avoid wasting compute on rejected models. Scenarios run only after agent approval (Phase 8).

**6 Model Types:**

| Type | Library | Purpose |
|------|---------|---------|
| OLS | statsmodels | Baseline with p-values |
| Ridge | scikit-learn | L2 — handles multicollinearity |
| Lasso | scikit-learn | L1 — automatic feature selection |
| ElasticNet | scikit-learn | Combined L1+L2 |
| Bayesian Ridge | scikit-learn | Uncertainty estimates |
| Huber | scikit-learn | Robust to outliers |

**Generates:** `model_rankings.png`, `best_model_diagnostics.png`, `scenarios.png`

---

### `narrative_generator.py` — AI Narrative Generation

Handles Phase 9. Uses GPT-4.1 with full pipeline context.

| Function / Method | Purpose |
|-------------------|---------|
| `get_llm_client()` | Initialize Azure OpenAI client |
| `call_llm(client, prompt, ...)` | Call GPT-4.1 |
| `NarrativeGenerator.narrate_eda(...)` | EDA executive summary |
| `NarrativeGenerator.narrate_outliers(...)` | Data cleaning explanation |
| `NarrativeGenerator.narrate_features(..., vif_summary, channel_ranking)` | Feature engineering with multicollinearity context |
| `NarrativeGenerator.narrate_modeling(..., vif_info, agent_context)` | Model results with convergence + agent iteration history |
| `NarrativeGenerator.narrate_scenarios(...)` | Scenario recommendations |
| `NarrativeGenerator.save_report(output_dir)` | Save all as markdown |
| `generate_all_narratives(result, ..., agent_context=)` | Convenience: extracts VIF, channel ranking, agent context and passes to all narrate methods |

**Context passed to narratives:**
- VIF summary per spec (from feature engineering)
- Channel ranking with correlations (from EDA correlation matrix)
- Multicollinearity explanation (why groups vs individual channels)
- Agent iteration context: how many iterations, quality progression, what adjustments were made
- Convergence-confirmed feature directions
- Model selection note (why only linear models, no tree-based)

**Generates:** `outputs/reports/mmix_narrative_report.md`

---

### `agent_orchestrator.py` — Agentic Pipeline & State Machine

The orchestration layer. Single pipeline for all runs.

**Classes:**

| Class | Purpose |
|-------|---------|
| `AgentState` | Full pipeline state: data, results, reasoning trace, quality scores, `spec_strategy`, `adjustments` |
| `QualityEvaluator` | Three-layer evaluation: rule-based HARD + LLM with EDA context + decision logic |

**Pipeline Tools:**

| Tool | Phase | Critical? |
|------|-------|-----------|
| `tool_load_data()` | 1 | Yes |
| `tool_run_eda()` | 2 (uses pre-loaded data) | No |
| `tool_run_outliers()` | 3 | Yes |
| `tool_run_aggregation()` | 4 | Yes |
| `tool_run_features()` | 5 (applies spec strategy filter) | Yes |
| `tool_run_modeling(skip_scenarios=True)` | 6 | Yes |
| `evaluate_and_decide()` | 7 | Agentic |
| `tool_run_scenarios()` | 8 (post-loop, approved model only) | Post-loop |
| `tool_run_narratives()` | 9 (with agent_context) | Post-loop |

**Spec Strategy (set by agent):**

| Strategy | Specs Used | When |
|----------|-----------|------|
| `base` | A-F (6 specs, 3 features) | Iteration 1 (conservative start) |
| `expanded` | G-J only (4 specs, 6-8 features) | Iteration 2 (delta — only new specs) |
| `groups` | A, J (2 specs) | If VIF issues detected |
| `monthly` | A-F on monthly data | Iteration 3 (granularity fallback, re-aggregates) |

**Quality Evaluation — Three Layers:**

| Layer | Type | Can Override? |
|-------|------|--------------|
| Rule-based HARD | R²≥0.50, Adj R²≥0.45, ordinality, CV stability | No — deterministic |
| LLM (GPT-4.1) | HARD/SOFT criteria + EDA context + convergence | Receives data-driven findings |
| Decision logic | Rule FAIL→RETRY always; LLM RETRY respected; max iter→PROCEED | Code-enforced |

**Model Comparison:** After iteration 2+, agent compares current best R² vs previous best and keeps the winner.

**Generates:** `outputs/reports/agent_reasoning_trace.txt`

---

## Data Files

| File | Key | Rows | Columns | Description |
|------|-----|------|---------|-------------|
| `firstfile.csv` | transactions | 1,578,079 | 9 | Transaction-level: product, category, date, GMV, units, discount, MRP |
| `Sales.csv` | sales | 1,048,575 | 13 | Sales: delivery/order dates, status, quantities |
| `SecondFile.csv` | monthly | 12 | 39 | Monthly aggregated: GMV, units, 9 channel spends, NPS, discount |
| `SpecialSale.csv` | special_sales | 44 | 2 | Sale event dates (Diwali, Eid, Independence Day, etc.) |
| `MediaInvestment.csv` | investment | 12 | 12 | Monthly media investment by channel (cross-validation) |
| `MonthlyNPSscore.csv` | nps | 12 | 2 | Monthly Net Promoter Scores |
| `ProductList.csv` | products | 75 | 3 | Product catalog: 75 products, 5 categories |

---

## Output Files

### Plots (outputs/plots/)

| File | Generated By | Phase |
|------|-------------|-------|
| `correlation_analysis.png` | eda_pipeline.py | 2 |
| `nps_revenue.png` | eda_pipeline.py | 2 |
| `channel_overlap.png` | eda_pipeline.py | 2 |
| `category_breakdown.png` | eda_pipeline.py | 2 |
| `special_sale_impact.png` | eda_pipeline.py | 2 |
| `media_investment.png` | eda_pipeline.py | 2 |
| `national_trends.png` | eda_pipeline.py | 2 |
| `outlier_summary.png` | outlier_detection.py | 3 |
| `feature_distributions.png` | feature_engineering.py | 5 |
| `features_vs_target.png` | feature_engineering.py | 5 |
| `model_rankings.png` | modeling_engine.py | 6 |
| `best_model_diagnostics.png` | modeling_engine.py | 6 |
| `scenarios.png` | modeling_engine.py | 8 (post-loop) |

### Reports (outputs/reports/)

| File | Generated By | Contents |
|------|-------------|---------|
| `mmix_narrative_report.md` | narrative_generator.py | AI-generated report: 5 sections + agent iteration context |
| `agent_reasoning_trace.txt` | agent_orchestrator.py | Timestamped decision log with quality progression per iteration |
