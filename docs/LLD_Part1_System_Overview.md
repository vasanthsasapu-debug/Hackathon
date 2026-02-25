# Agentic MMIX Pipeline — Low-Level Design & System Documentation

# Part 1: System Overview & Execution Flow

---

## 1. Problem Statement & Scope

### Problem

Marketing Mix Modeling (MMIX) in e-commerce is traditionally a slow, manual, retrospective exercise. Analysts spend weeks cleaning data, running models, and preparing decks — by which time the insights are too late to influence in-quarter budget decisions. The typical workflow involves:

- **Data collection and cleaning** (1-2 weeks): Gathering spend data from multiple platforms, reconciling transaction records, handling missing values and outliers.
- **Feature engineering** (3-5 days): Creating log transforms, lag variables, promotional features — typically hardcoded to the analyst's past experience.
- **Model building** (1-2 weeks): Running a handful of regression specs, manually checking coefficient signs, iterating on feature selection.
- **Report generation** (3-5 days): Translating statistical output into business language for marketing directors and C-suite.

Total cycle time: **3-6 weeks** from data availability to actionable insight. By then, the budget window has shifted.

### Solution

An **Agentic MMIX Workflow** that automates the entire lifecycle: data ingestion, EDA, outlier detection, feature engineering, modeling, scenario simulation, and business narrative generation. The key differentiator is the **agentic loop** — the system evaluates its own output quality using a combination of rule-based HARD criteria and GPT-4.1 reasoning, and iterates with progressively richer feature strategies until quality criteria are met.

The agent doesn't just run a pipeline — it **reasons about the results**, detects problems (low R², multicollinearity, ordinality violations), suggests specific improvements (expand to group-based specs, switch granularity), and applies those improvements automatically. Every decision is logged with a timestamp and rationale in a reasoning trace for full transparency.

### Scope

| Dimension | Detail |
|-----------|--------|
| **Data** | 12 months of e-commerce data (Jul 2015 – Jun 2016) |
| **Products** | 5 categories: Camera, CameraAccessory, EntertainmentSmall, GameCDDVD, GamingHardware |
| **Channels** | 9 media channels → 4 EDA-validated groups |
| **Granularity** | Weekly (47 data points) and Monthly (10 data points) |
| **Models** | 6 linear algorithm types × 10 feature specifications × 2 transform variants |
| **Max candidates** | 120 per iteration (72 base + 48 expanded delta) |
| **Agentic** | 3-iteration loop: base → expanded → monthly fallback |
| **LLM** | Azure OpenAI GPT-4.1 for quality evaluation and narrative generation |
| **Output** | Ranked models, elasticities, 11+ scenario simulations, AI-generated business narratives |

### Why Weekly/Monthly — Not Daily or Transaction-Level

The transaction file contains 1.6M individual orders — enough for daily aggregation. However, marketing spend data (MediaInvestment.csv, SecondFile.csv) is only available at **monthly granularity** (12 rows of channel-level spend). You cannot attribute daily GMV to daily spend when spend is only reported monthly — assigning the same monthly figure to every day creates artificial patterns and violates the independence assumption that regression requires.

Weekly aggregation (47 data points) is the sweet spot: 4× the sample size of monthly while maintaining meaningful variation in marketing variables. Monthly (10 data points) is supported but severely limits model complexity to 2-3 predictors.

Daily-level MMIX with daily spend data (from ad platform APIs) would be a natural production enhancement, enabling adstock/carry-over transformations and significantly improving model power.

| Granularity | Data Points | Feasible? | Rationale |
|-------------|-------------|-----------|-----------|
| Transaction | 1.6M | No | No transaction-level spend data; different paradigm (MTA) |
| Daily | ~365 | No | Marketing spend only monthly; violates independence |
| **Weekly** | **47** | **Yes (primary)** | Best balance of sample size and spend variation |
| Monthly | 10 | Yes (fallback) | Limited complexity; agent uses as iteration 3 fallback |

---

## 2. Assumptions (A1–A9)

All assumptions are documented in `outlier_detection.py → ASSUMPTIONS` dict and printed at pipeline start. Each includes description, decision, and detailed reasoning.

| ID | Assumption | Decision | Reasoning |
|----|-----------|----------|-----------|
| A1 | Aug 2015 shows <5% of median GMV (0.04 Cr vs ~39 Cr median) | **EXCLUDE** | Platform launched Jul 2015. Aug is ramp-up — including it would distort elasticity estimates. Losing 1 of 12 months is costly, but keeping an anomalous point is worse. |
| A2 | Discount spikes during sales (30-40% of MRP) during Diwali, Eid, Independence Day | **KEEP** | Standard e-commerce promotional behavior. Removing would eliminate the promotional signal we need to measure. Created `sale_intensity` feature instead. |
| A3 | Radio/Other channels have zero spend in several months | **KEEP zeros** | Legitimate budget decisions (not all channels active every month). `log(x+1)` transform handles zeros cleanly. |
| A4 | NPS has -0.96 correlation with GMV | **KEEP as seasonality proxy** | High-sale periods attract deal-seeking buyers who rate lower (delivery delays, stock-outs). NPS absorbs seasonal variance that `sale_flag` alone cannot capture. Coefficient = "seasonal adjustment", NOT "NPS effect." Removing NPS drops R² significantly. Revisit with 24+ months. |
| A5 | Oct/Sep 2015 high GMV coincides with festival season | **KEEP** | Festival peaks are the core pattern being modeled. `sale_flag` captures the effect. |
| A6 | Discount may be endogenous (company discounts more when sales are low) | **Use as CONTROL only** | Not a primary MMIX lever. Included in specs as `discount_intensity` (ratio: discount/MRP) to control for promotional depth without implying causality. |
| A7 | Only 12 months of data limits model complexity | **Weekly aggregation preferred** | Increases n from 10 to 47. Monthly used as iteration 3 fallback. Max predictors: 3 for monthly, 6-8 for weekly. |
| A8 | Sales.csv GMV column is dtype `object` | **Coerce to numeric** | Data quality issue at source. Handled in `load_all_data()` post-load fixes. |
| A9 | Radio (-0.95) and Other (-0.96) show strong negative GMV correlation | **Allow negative coefficients** | Sparse activity in low-GMV months creates inverse pattern. Not causal. Convergence confirms negative direction across all models. Ordinality check exempts these channels. |

---

## 3. Key Findings

1. **Sale events are the dominant GMV driver** — activating a sale event lifts GMV by +45-50%, far exceeding any media spend change. This is the single most important lever.

2. **Digital performance is the strongest channel group** — Online.marketing + Affiliates (corr: +0.88 with GMV). Sponsorship is the strongest individual traditional channel.

3. **Digital brand shows consistently negative marginal impact** — Digital + Content.Marketing + SEM has positive raw correlation (+0.51) but convergence across all models confirms negative coefficients when controlling for other spend. Likely awareness spend with delayed ROI not captured in 12 months.

4. **Budget reallocation shows limited impact** — Shifting 10% from TV to Online.marketing yields only +0.7% GMV. Growth comes from increasing total spend and running sale events, not from reallocation.

5. **Ridge/Huber regression consistently outperforms** — L2 regularization handles the multicollinearity between channel groups effectively. Huber adds outlier robustness for sale-period spikes.

6. **NPS is a seasonality proxy, not a satisfaction metric** — Its -0.96 GMV correlation is an artifact of high-sale periods attracting lower-satisfaction buyers. Including it as a control improves R² significantly.

7. **Convergence-based ordinality is more reliable than fixed rules** — Traditional MMIX requires all spend coefficients ≥ 0, but this fails for channels with legitimately negative marginal impact. Using cross-model consensus as the truth produces better rankings.

---

## 4. Architecture Overview

### Single Entry Point

All runs — whether invoked from `main.py` or `agent_orchestrator.py` — go through `run_agentic_pipeline()`. There is no separate "standard" pipeline. If the model passes quality checks on iteration 1, the agentic loop exits immediately with zero overhead.

```
main.py (CLI wrapper)
 │
 ├── -g weekly → run_agentic_pipeline(granularity="weekly")
 ├── -g monthly → run_agentic_pipeline(granularity="monthly")
 └── -g both → run_agentic_pipeline() × 2 + compare
```

### Module Map

```
Entry Point:
└── main.py ─────────── CLI flags, both-granularity comparison

Shared Modules (src/):
├── config.py ────────── Paths, channels, groups, model settings, LLM config
├── eda_pipeline.py ──── Data loading + 11 EDA analyses + correlation intelligence
├── outlier_detection.py  Cleaning, validation, 9 assumptions (A1-A9)
├── data_aggregation.py ─ Weekly/monthly aggregation with sale mapping
├── feature_engineering.py  Transforms, 10 data-driven specs, seasonality features
├── modeling_engine.py ── 6 model types, convergence-based ordinality, scenarios
├── narrative_generator.py  GPT-4.1 narratives with VIF/correlation/agent context
└── agent_orchestrator.py  Agentic state machine, 3-layer evaluator, iterative loop

Output:
├── outputs/plots/ ────── 13 PNG visualizations
└── outputs/reports/ ──── Narrative report (MD), reasoning trace (TXT)
```

### Design Decision: Custom Orchestrator vs LangChain/CrewAI

The orchestrator uses a custom state machine rather than LangChain or CrewAI because:

1. The pipeline has a well-defined linear flow with one decision point (post-modeling evaluation). LangChain adds overhead for what is essentially: run steps → evaluate → conditionally loop.
2. The custom approach gives full control over the reasoning trace format — every decision is recorded with timestamp, phase, iteration, and details dict.
3. No external dependencies beyond the OpenAI SDK.
4. The architecture is LangGraph-compatible: each `tool_*` function maps to a LangGraph node, and `evaluate_and_decide` is the conditional edge. Migration would require wrapping functions as LangGraph tools and defining the state graph.

---

## 5. Execution Flow

### 5.1 Agentic Pipeline (all runs)

```
run_agentic_pipeline()
 │
 ├── Initialize AgentState(granularity, spec_strategy="base")
 ├── Initialize QualityEvaluator(llm_client)
 │
 ├── [PHASE 1] tool_load_data() → state.data (7 DataFrames)
 │    └── eda_pipeline.load_all_data()
 │
 ├── [PHASE 2] tool_run_eda(data=state.data) → state.corr_matrix
 │    └── eda_pipeline.run_full_eda(data=pre_loaded)  ← No double loading
 │    └── Correlation intelligence: feature signs, inter-channel, groups, NPS
 │
 ├── [PHASE 3] tool_run_outliers() → state.clean_data, state.outlier_log, state.assumptions
 │    └── outlier_detection.run_outlier_pipeline()
 │    └── 9 assumptions printed and returned
 │
 ├── [PHASE 4] tool_run_aggregation() → state.aggregated_data
 │    └── data_aggregation.build_modeling_dataset()
 │    └── Agent can re-aggregate to monthly on iteration 3
 │
 ├── AGENTIC LOOP (max 3 iterations):
 │    │
 │    │  ┌─ Iteration 1: strategy="base" ─────────────────────────────┐
 │    │  │  Specs A-F (6 specs, 3 features each)                      │
 │    │  │  6 × 6 models × 2 transforms = 72 candidates              │
 │    │  └────────────────────────────────────────────────────────────┘
 │    │
 │    │  ┌─ Iteration 2: strategy="expanded" (delta) ────────────────┐
 │    │  │  Specs G-J ONLY (4 specs, 6-8 features, group-based)      │
 │    │  │  4 × 6 × 2 = 48 NEW candidates (not re-running A-F)      │
 │    │  │  Compare best vs iter 1 best → keep winner                │
 │    │  └────────────────────────────────────────────────────────────┘
 │    │
 │    │  ┌─ Iteration 3: strategy="monthly" (fallback) ──────────────┐
 │    │  │  Re-aggregate to monthly (10 data points)                  │
 │    │  │  Specs A-F with monthly threshold (R² ≥ 0.70)             │
 │    │  │  6 × 6 × 2 = 72 candidates on different granularity      │
 │    │  └────────────────────────────────────────────────────────────┘
 │    │
 │    ├── [PHASE 5] tool_run_features(state)
 │    │    ├── run_feature_engineering() → full matrix
 │    │    └── Filter specs by state.spec_strategy (base/expanded/groups/monthly)
 │    │
 │    ├── [PHASE 6] tool_run_modeling(skip_scenarios=True)
 │    │    ├── Build all models
 │    │    ├── Initial rank by composite score
 │    │    ├── Compute convergence (cross-model consensus)
 │    │    ├── Re-score ordinality using convergence as truth (two-pass)
 │    │    └── Re-rank → final best model for this iteration
 │    │
 │    ├── [PHASE 7] evaluate_and_decide(state, evaluator)
 │    │    ├── Layer 1: Rule-based HARD criteria (R² ≥ 0.50, Adj R² ≥ 0.45, etc.)
 │    │    ├── Layer 2: LLM evaluation (GPT-4.1 with EDA context + HARD/SOFT criteria)
 │    │    ├── Layer 3: Decision logic (LLM RETRY respected; rule FAIL always RETRY)
 │    │    └── Returns "PROCEED" or "RETRY"
 │    │
 │    └── If RETRY:
 │         ├── suggest_improvements(quality, state, llm_verdict)
 │         ├── Set state.spec_strategy based on issues
 │         └── If iter 3: re-aggregate to monthly
 │
 ├── [PHASE 8] tool_run_scenarios() → on approved model ONLY
 │    └── build_scenario_simulator() + run_standard_scenarios()
 │    └── 11+ what-if simulations + plot
 │
 ├── [PHASE 9] tool_run_narratives(agent_context)
 │    └── generate_all_narratives() with iteration history, quality progression, adjustments
 │    └── 5 section narratives + markdown report
 │
 └── Save reasoning trace → agent_reasoning_trace.txt
```

### 5.2 Both-Granularity Comparison (main.py -g both)

When invoked with `-g both`, `main.py` runs `run_agentic_pipeline()` independently for weekly and monthly, then calls `_compare_granularities()` which prints:

- Side-by-side: best spec, model type, R², Adj R², CV R², composite, ordinality, iterations, strategy
- Coefficients comparison
- Insight agreement: for each feature present in both, whether weekly and monthly agree on direction (POSITIVE/NEGATIVE/MIXED)

---

## 6. File-by-File Purpose

### 6.1 config.py

**Purpose:** Single source of truth for all configuration. All other modules import from here. No hardcoded paths or constants elsewhere.

**Key exports:**

| Export | Type | Description |
|--------|------|-------------|
| `get_paths(data_dir, output_dir)` | Function → Dict | Resolves root, data, output, plots, reports directories |
| `DATA_FILES` | Dict | Registry of 7 CSV files with filenames, separators, date columns, format overrides |
| `MEDIA_CHANNELS` | List[str] | 9 channel names in order |
| `CHANNEL_GROUPS` | Dict[str, List] | 4 EDA-validated groups. SEM in digital_brand (not digital_perf) based on correlation evidence. Includes rationale comment. |
| `LOG_TO_RAW_MAP` | Dict[str, str] | Maps log-transformed column names back to raw names for scenario simulation |
| `MODEL_SETTINGS` | Dict | Composite score weights (35/25/25/15), VIF threshold (10), ordinality flag, max predictors, **quality_thresholds** (min_r2, min_adj_r2, max_vif, min_models_passed, min_ordinality_rate) |
| `LLM_CONFIG` | Dict | Azure OpenAI config read from environment variables (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_OPENAI_API_KEY`) |
| `PipelineSummary` | Class | Collects structured summaries from every pipeline step. `get_full_summary()` produces the text consumed by LLM narrative generation. |
| `find_col(df, candidates)` | Function → str | Returns first column from candidates that exists in df |
| `get_channel_cols(df)` | Function → List | Returns MEDIA_CHANNELS that exist in df |
| `logger` | Logger | Configured logging instance used by all modules |

**Channel Groups (with EDA rationale):**

```python
CHANNEL_GROUPS = {
    "traditional":          ["TV", "Sponsorship"],
    "digital_performance":  ["Online.marketing", "Affiliates"],
    "digital_brand":        ["Digital", "Content.Marketing", "SEM"],
    "other":                ["Radio", "Other"],
}
# SEM moved from digital_performance to digital_brand based on
# EDA correlation evidence (SEM↔Digital r=0.97, SEM↔Content.Mkt r=0.96)
```

### 6.2 eda_pipeline.py

**Purpose:** Load all 7 datasets, auto-classify columns, validate data quality, produce 11 EDA analyses with 7 visualizations. Includes **correlation intelligence** that feeds into the LLM evaluation prompt.

**Key design decision:** `run_full_eda()` accepts an optional `data=` parameter. When the agentic pipeline calls it, pre-loaded data is passed, avoiding the 1.6M+ row double-load that occurred in the original implementation.

**Correlation intelligence (new):** The `correlation_analysis()` function now produces:
- Feature-GMV sign classification (which channels are positive vs negative)
- Inter-channel correlations flagging pairs with |r| > 0.85 as grouping candidates
- Channel GROUP-level correlations (validates that groups aren't too correlated for regression)
- NPS seasonality detection (flags if NPS-GMV correlation < -0.5)
- NPS-channel correlations

These findings are stored on `state.corr_matrix` and dynamically extracted by the LLM evaluation prompt — the LLM learns about NPS, negative channels, and multicollinearity from the data, not from hardcoded instructions.

**Key outputs:** data dict, classifications dict, issues list, correlation matrix DataFrame

**Generates:** 7 PNG plots (correlation_analysis, nps_revenue, channel_overlap, category_breakdown, special_sale_impact, media_investment, national_trends)

### 6.3 outlier_detection.py

**Purpose:** Clean transaction and monthly data, detect statistical outliers (IQR + Z-score), validate business context (sale vs non-sale), check cross-dataset reconciliation. Documents 9 assumptions (A1-A9).

**Key design decisions:**
- Aug 2015 excluded automatically if GMV < 5% of median (A1)
- Discount spikes during sales are kept and featurized, not removed (A2)
- Zero-spend months are kept as legitimate budget decisions (A3)
- NPS values retained despite negative correlation (A4)

**Key outputs:** clean_data dict, cleaning log list, ASSUMPTIONS dict (returned and printed)

**Generates:** outlier_summary.png

### 6.4 data_aggregation.py

**Purpose:** Build modeling dataset at weekly or monthly granularity. Weekly path aggregates 1.58M transactions into 47 weeks and distributes monthly channel spend evenly across weeks. Maps 44 sale events to weeks. Joins monthly NPS to weekly via year-month matching.

**Key design decision:** The agent can trigger re-aggregation to monthly on iteration 3 by calling `tool_run_aggregation()` again after setting `state.granularity = "monthly"`.

**Key outputs:** {"data": DataFrame, "n_periods": int, "summary": str}

### 6.5 feature_engineering.py

**Purpose:** Transform raw data into modeling features. Creates log transforms (12 variables), sale features, channel groups (4 EDA-validated groups), discount features, lag features, NPS (as seasonality proxy), and seasonality features (available but not in specs). Builds 10 curated model specifications with data-driven channel ranking. Checks VIF per spec.

**Key design decisions:**
- Channel ranking is data-driven: ranked by |correlation| with log_total_gmv, not hardcoded
- Weekly specs use channel GROUPS, not individual channels (EDA showed inter-channel r=0.97-0.99)
- NPS kept as seasonality proxy despite negative coefficient (removing drops R² significantly)
- Seasonality features (is_festival_season, month_sin/cos) created but not in specs — reserved for future use
- 10 specs curated to test distinct hypotheses without redundancy

**Key outputs:** feature_matrix DataFrame (47×46 weekly or 10×24 monthly), feature_sets dict (6 base + 4 weekly = 10 specs), VIF results per spec

**Generates:** feature_distributions.png, features_vs_target.png

### 6.6 modeling_engine.py

**Purpose:** Build, evaluate, rank models. 6 linear algorithm types × 10 specs × 2 transforms = up to 120 candidates per iteration. Implements convergence-based ordinality (two-pass ranking). Scenario simulation with raw channel name translation. Supports `skip_scenarios=True` for the agentic loop.

**Key design decisions:**
- Tree-based models (XGBoost, Random Forest) excluded: they don't produce interpretable elasticities essential for marketing budget recommendations
- Huber regression added for outlier robustness (sale-period spikes)
- Ordinality only enforces positive on unambiguous features (sale_flag, Total.Investment). Channel/group features validated against convergence consensus.
- Two-pass ranking: build → initial rank → convergence → re-rank with convergence truth
- `skip_scenarios` flag: during the agentic loop, scenarios are skipped to avoid wasting compute on rejected models

**Key outputs:** ranked_models list, best_model dict, convergence dict, scenarios list (post-loop), simulator function

**Generates:** model_rankings.png, best_model_diagnostics.png, scenarios.png (post-loop only)

### 6.7 narrative_generator.py

**Purpose:** Generate business-ready narratives using GPT-4.1 for each pipeline step. Produces 5-section markdown report. Receives VIF data, channel correlation ranking, and agent iteration context.

**Key design decisions:**
- Feature narrative receives VIF summary and channel ranking — explains multicollinearity and why groups are used
- Modeling narrative receives agent_context (iterations, quality progression, adjustments) — mentions if the agent iterated
- All context is dynamically extracted from pipeline results, not hardcoded
- System prompt: Senior marketing analytics consultant — "What happened → Why it matters → What to do"

**Key outputs:** NarrativeGenerator instance with narratives dict, saved markdown report

**Generates:** outputs/reports/mmix_narrative_report.md

### 6.8 agent_orchestrator.py

**Purpose:** Agentic pipeline that runs the MMIX workflow with LLM-driven quality evaluation, iterative loop-back with progressive strategy escalation, and full reasoning trace. Single pipeline for all runs.

**Key design decisions:**
- `AgentState` tracks `spec_strategy` ("base"/"expanded"/"groups"/"monthly") and `adjustments`
- `tool_run_features()` filters specs based on agent strategy — delta approach on iteration 2
- `tool_run_modeling(skip_scenarios=True)` skips scenarios during loop
- `tool_run_scenarios()` runs post-loop on approved model only
- `tool_run_narratives()` passes full agent context (iterations, quality scores, adjustments)
- Three-layer quality evaluation: rule-based HARD + LLM with dynamic EDA context + decision logic
- LLM prompt includes HARD criteria that "MUST be RETRY" and SOFT criteria for judgment
- Model comparison after iteration 2+: keeps whichever model (current vs previous) has higher R²
- Monthly fallback on iteration 3: re-aggregates data and runs base specs with R² ≥ 0.70 threshold

**Key outputs:** AgentState with all results, reasoning trace, quality scores per iteration

**Generates:** outputs/reports/agent_reasoning_trace.txt

---

## 7. Data Flow Diagram

```
firstfile.csv (1.58M rows)
 │
 ├─ [weekly path] → aggregate to 47 weeks → distribute channel spend → map 44 sale events → add NPS
 │                                                                                          │
Sales.csv (1.05M rows)                                                                      │
 │  (EDA only)                                                                              │
 │                                                                                          ▼
SecondFile.csv (12 rows) → clean monthly (excl Aug 2015) → [monthly path] ───→ Feature Matrix
 │                                                                              (47×46 weekly
 │  (channel spend source for weekly distribution)                               or 10×24 monthly)
 │                                                                                    │
SpecialSale.csv (44 events) → map to weeks/months → sale_flag, sale_days              │
 │                                                                                    ▼
MonthlyNPSscore.csv (12 rows) → repeat per week → NPS, nps_standardized        10 Model Specs
 │                                                                              × 6 Model Types
MediaInvestment.csv (12 rows) → reconciliation check only                       × 2 Transforms
 │                                                                                    │
ProductList.csv (75 products) → reference only                                        ▼
                                                                               Convergence-Based
                                                                               Two-Pass Ranking
                                                                                      │
                                                                                      ▼
                                                                               Agent Evaluation
                                                                               (HARD + LLM + Logic)
                                                                                      │
                                                                          ┌───────────┴───────────┐
                                                                          ▼                       ▼
                                                                       PROCEED                  RETRY
                                                                          │                 (adjust strategy,
                                                                          ▼                  loop back)
                                                                    Scenario Simulation
                                                                    (approved model only)
                                                                          │
                                                                          ▼
                                                                    AI Narratives
                                                                    (with agent context)
                                                                          │
                                                                          ▼
                                                                    📦 Final Outputs
                                                                    13 plots + reports + trace
```

---

## 8. Output Artifacts

| Artifact | Location | Format | Generated By | Phase |
|----------|----------|--------|-------------|-------|
| Correlation analysis | outputs/plots/correlation_analysis.png | PNG | eda_pipeline.py | 2 |
| NPS vs revenue | outputs/plots/nps_revenue.png | PNG | eda_pipeline.py | 2 |
| Channel overlap | outputs/plots/channel_overlap.png | PNG | eda_pipeline.py | 2 |
| Category breakdown | outputs/plots/category_breakdown.png | PNG | eda_pipeline.py | 2 |
| Special sale impact | outputs/plots/special_sale_impact.png | PNG | eda_pipeline.py | 2 |
| Media investment | outputs/plots/media_investment.png | PNG | eda_pipeline.py | 2 |
| National trends | outputs/plots/national_trends.png | PNG | eda_pipeline.py | 2 |
| Outlier summary | outputs/plots/outlier_summary.png | PNG | outlier_detection.py | 3 |
| Feature distributions | outputs/plots/feature_distributions.png | PNG | feature_engineering.py | 5 |
| Features vs target | outputs/plots/features_vs_target.png | PNG | feature_engineering.py | 5 |
| Model rankings | outputs/plots/model_rankings.png | PNG | modeling_engine.py | 6 |
| Best model diagnostics | outputs/plots/best_model_diagnostics.png | PNG | modeling_engine.py | 6 |
| Scenario results | outputs/plots/scenarios.png | PNG | modeling_engine.py | 8 (post-loop) |
| Narrative report | outputs/reports/mmix_narrative_report.md | Markdown | narrative_generator.py | 9 |
| Reasoning trace | outputs/reports/agent_reasoning_trace.txt | Text | agent_orchestrator.py | Final |
