# Agentic Marketing Mix Modeling (MMIX) Pipeline

## Overview

A complete, production-ready **Agentic MMIX workflow** for pharmaceutical and marketing teams. The pipeline automatically:

1. **Analyzes** promotional channel performance (national + segment level)
2. **Models** sales response to each channel (with multiple model types)
3. **Optimizes** budget allocation across 4 scenarios
4. **Narrates** findings using Azure OpenAI GPT-4.1

---

## Architecture

```
src/
├── 1_core/                          # Component 1: End-to-end modeling
│   ├── column_classifier.py         # Auto-classify column semantics
│   ├── eda_enhanced.py              # National + segment-level EDA
│   ├── outlier_detection.py         # [EXISTING] Enhanced with segment support
│   ├── feature_engineering.py       # [EXISTING] Enhanced with response curve prep
│   ├── modeling_enhanced.py         # GLM, Fixed/Random Effects, ranking
│   ├── response_curves.py           # Extract channel elasticity & curves
│   └── __init__.py
│
├── 2_agentic/                       # Component 2: Agent orchestration
│   ├── orchestrator.py              # LangGraph-style state machine
│   ├── llm_integration.py           # Step-specific GenAI narratives
│   ├── agent_nodes.py               # [Optional] Expanded node definitions
│   └── __init__.py
│
├── 3_polish/                        # Component 3: Delivery & advanced features
│   ├── mix_optimizer.py             # 4-scenario budget optimization
│   ├── export_excel.py              # Multi-sheet Excel workbook
│   ├── export_ppt.py                # Professional PowerPoint slides
│   ├── goal_optimizer.py            # NLP-based goal parsing
│   └── __init__.py
│
├── llm_config.py                    # [EXISTING] Azure OpenAI config
└── main.py                          # Entry point (CLI + programmatic)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Azure OpenAI

Create a `.env` file in the project root:

```env
AZURE_ENDPOINT="https://zs-eu1-ail-agentics-openai-team10.openai.azure.com/"
API_VERSION="2024-12-01-preview"
DEPLOYMENT_NAME="gpt-4.1"
EMBEDDING_DEPLOYMENT="text-embedding-3-small"
AZURE_OPENAI_KEY="your_api_key_here"
```

### 3. Run the Pipeline

```bash
python main.py --data-dir data --output-dir outputs
```

**Options:**
- `--data-dir`: Path to data folder (default: `data`)
- `--output-dir`: Path for results (default: `outputs`)
- `--no-llm`: Skip GenAI narratives (for testing)
- `--no-excel`: Skip Excel export
- `--no-ppt`: Skip PowerPoint export

---

## Component Details

### Component 1: Core Pipeline

**Purpose:** All modeling work before agentic features.

**Modules:**

| Module | Purpose | Key Functions |
|--------|---------|---|
| `column_classifier.py` | Auto-assign semantic categories | `classify_columns()`, `validate_classification()` |
| `eda_enhanced.py` | National + segment EDA | `run_segment_eda()`, `calculate_reach_frequency_engagement()` |
| `response_curves.py` | Extract elasticity & curves | `run_response_curve_extraction()`, `calculate_elasticity()` |
| `modeling_enhanced.py` | Extended models + ranking | `fit_glm_poisson()`, `rank_models()`, `visualize_model_comparison()` |

**Example:**

```python
from src.1_core import run_column_classification, run_segment_eda

# Classify columns
classification = run_column_classification(df, verbose=True)

# Run segment-level EDA
eda_results = run_segment_eda(
    monthly_df,
    channel_columns=["TV", "Digital", "SEM"],
    segment_column="Product_Category"
)
```

---

### Component 2: Agentic Wrapper

**Purpose:** Orchestrate all steps with feedback loops and LLM integration.

**Modules:**

| Module | Purpose | Key Classes |
|--------|---------|---|
| `orchestrator.py` | State machine & workflow | `Orchestrator`, `PipelineState` |
| `llm_integration.py` | Step-specific narratives | `generate_eda_narrative()`, `get_agent_critique()` |

**Features:**
- ✅ Automatic feedback loops (e.g., "VIF too high → re-run feature engineering")
- ✅ GenAI summaries after each step
- ✅ State tracking & error handling

**Example:**

```python
from src.2_agentic import Orchestrator, get_llm_client

llm_client = get_llm_client()
orchestrator = Orchestrator(data, llm_client)
state = orchestrator.run_full_pipeline(auto_feedback_loop=True)

# Check results
print(f"EDA narrative: {state.eda_narrative}")
print(f"Top model: {state.best_model['type']}")
```

---

### Component 3: Polish

**Purpose:** Optimization, export, and advanced features.

**Modules:**

| Module | Purpose | Key Functions |
|--------|---------|---|
| `mix_optimizer.py` | 4-scenario optimization | `run_mix_optimization()`, `build_blue_sky_scenario()` |
| `export_excel.py` | Multi-sheet Excel | `export_to_excel()` |
| `export_ppt.py` | Professional PowerPoint | `generate_ppt_presentation()` |
| `goal_optimizer.py` | NLP goal parsing | `GoalParser()`, `run_goal_based_optimization()` |

**Optimization Scenarios:**

1. **Base Case:** Historical execution (no optimization)
2. **Budget Neutral:** Reallocate within fixed total spend
3. **Max Profit:** Maximize ROI with current budget
4. **Blue Sky:** Unconstrained profit maximization

**Example:**

```python
from src.3_polish import run_mix_optimization, export_to_excel

# Optimize
scenarios = run_mix_optimization(
    monthly_df,
    channel_columns=["TV", "Digital", "SEM"],
    elasticities={"TV": 0.5, "Digital": 0.8, "SEM": 0.6}
)

# Export
export_to_excel(
    "results.xlsx",
    eda_results=eda_results,
    scenarios=scenarios["scenarios"]
)
```

---

## End-to-End Example

```python
from src.1_core import run_column_classification, run_segment_eda
from src.2_agentic import Orchestrator, get_llm_client
from src.3_polish import run_mix_optimization, export_to_excel

# Step 1: Initialize
llm_client = get_llm_client()
data = {"monthly": df_monthly, "transactions": df_transactions}

# Step 2: Run full pipeline (agentic)
orchestrator = Orchestrator(data, llm_client)
state = orchestrator.run_full_pipeline()

# Step 3: Optimize
scenarios = run_mix_optimization(
    data["monthly"],
    channel_columns=state.state.features_engineered.columns,
    elasticities=state.elasticities
)

# Step 4: Export
export_to_excel(
    "final_report.xlsx",
    eda_results=state.eda_results,
    ranked_models=state.ranked_models,
    elasticities=state.elasticities,
    scenarios=scenarios["scenarios"]
)
```

---

## Data Requirements

| File | Purpose | Key Columns |
|------|---------|---|
| `Secondfile.csv` | Monthly aggregates | Date, total_gmv, channel spend (TV, Digital, SEM, etc.) |
| `Sales.csv` | Order-level details | Date, GMV, Product_Category, SLA |
| `firstfile.csv` | Daily transactions | Date, gmv_new, discount, units |
| `MediaInvestment.csv` | Channel spend | Date, TV, Digital, Sponsorship, etc. |
| `MonthlyNPSscore.csv` | Brand health | Month, NPS |
| `SpecialSale.csv` | Sale events | Date, Sales_name |
| `ProductList.csv` | Product master | Product_ID, Category |

---

## Output Files

After running the pipeline, you'll get:

```
outputs/
├── pipeline_state.json              # Pipeline execution summary
├── MMIX_Analysis_Results.xlsx       # All results in Excel
│   ├── EDA Summary
│   ├── Models (ranking, coefficients)
│   ├── Response Curves (elasticities)
│   ├── Optimization (4 scenarios)
│   └── Narratives (GenAI summaries)
└── MMIX_Analysis_Report.pptx        # Executive presentation
    ├── Title slide
    ├── EDA findings
    ├── Model rankings
    ├── Optimization scenarios
    └── Recommendations
```

---

## Advanced Features

### Goal-Based Optimization

Parse natural language goals and auto-apply constraints:

```python
from src.3_polish import run_goal_based_optimization

goals = "Increase Email ROI by 2%, keep TV spend below $50M"
result = run_goal_based_optimization(
    goals,
    base_allocation={"Email": 10e6, "TV": 40e6},
    current_metrics={"Email": 2.5, "TV": 3.0},
    elasticities={"Email": 0.7, "TV": 0.5}
)

print(result["applied_constraints"])
```

### Custom Model Families

The modeling engine supports:
- ✅ **Ridge & Bayesian Ridge** (baseline)
- ✅ **GLM Poisson** (count data)
- ✅ **GLM Gamma** (positive continuous)
- ✅ **Fixed Effects** (within-group variation)
- ✅ **Random Effects** (mixed models)

```python
from src.1_core import fit_glm_poisson, fit_glm_gamma

poisson_result = fit_glm_poisson(X, y)
gamma_result = fit_glm_gamma(X, y)
```

### Feedback Loops

The orchestrator automatically detects issues and re-runs steps:

```python
orchestrator = Orchestrator(data, llm_client)
state = orchestrator.run_full_pipeline(auto_feedback_loop=True)

# If VIF > 8 or R² < 0.5: auto-re-run feature engineering + modeling
```

---

## Model Ranking

Models are ranked on a composite score:

$$\text{Score} = 0.5 \times R^2 + 0.3 \times \text{Stability} + 0.2 \times \text{Ordinality}$$

Where:
- **Fit (R²):** Model accuracy (0-1)
- **Stability:** Low residual variance (0-1)
- **Ordinality:** Important channels have positive coefficients (0-1)

Top-10 models are exported with narratives explaining why each ranked high.

---

## GenAI Narratives

The system generates automated narratives for:

1. **EDA Step:** "Top 3 channels by reach are... with correlation to sales of..."
2. **Outlier Removal:** "Removed 2 outlier months because... This improves model fit by X%"
3. **Feature Engineering:** "Combined Digital + SEM due to 0.87 correlation... Rationale: reduce multicollinearity"
4. **Model Performance:** "Top model is Ridge regression with R²=0.78... Strengths: stable coefficients..."
5. **Optimization:** "Budget Neutral scenario shifts $10M from TV to Digital, expecting +12% ROI lift..."

---

## Configuration

### LLM Settings

Edit `src/llm_config.py` or `.env`:

```python
AZURE_ENDPOINT = "https://..."
API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4.1"
```

### Optimization Hyperparameters

In `src/3_polish/mix_optimizer.py`:

```python
# Budget flexibility (how much channels can shift)
flexibility = 1.2  # ±20% allowed

# Max spend per channel (prevent concentration)
max_single_channel = 0.3  # 30% of total budget
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Azure OpenAI API key not found` | Create `.env` file with `AZURE_OPENAI_KEY` |
| `openpyxl not installed` | `pip install openpyxl` |
| `python-pptx not installed` | `pip install python-pptx` |
| `Data files not found` | Ensure `data/` folder exists with CSV files |
| `LLM narratives are empty` | Check Azure API key and connection |
| `Model fit is poor (R² < 0.3)` | Review feature engineering; try different channel groups |

---

## Next Steps

1. **Customize data mapping:** Edit `column_classifier.py` patterns for your data
2. **Tune model constraints:** Adjust ordinality channels in `modeling_enhanced.py`
3. **Add custom scenarios:** Extend `mix_optimizer.py` with domain-specific logic
4. **Implement feedback actions:** Add decision rules in `orchestrator.py`
5. **Integrate with dashboards:** Export state JSON and feed to BI tools

---

## References

- **Marketing Mix Modeling:** https://en.wikipedia.org/wiki/Marketing_mix_modeling
- **Azure OpenAI:** https://learn.microsoft.com/en-us/azure/cognitive-services/openai/
- **Elasticity:** https://en.wikipedia.org/wiki/Elasticity_(economics)

---

## License & Support

This is a hackathon project built for commercial applications in pharma and marketing. Adapt as needed for your use case.

**Last Updated:** Feb 2026
