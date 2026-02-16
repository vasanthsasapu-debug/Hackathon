# Quick Reference Guide

## One-Liner Rundowns

### Component 1: Core Pipeline
**What:** End-to-end modeling (EDA → Features → Modeling)  
**Why:** All the statistical heavy lifting  
**When:** Always first  
**How:** `orchestrator.py` calls these automatically

### Component 2: Agentic Wrapper
**What:** Orchestration + LLM narratives + feedback loops  
**Why:** Automates the entire workflow, generates insights  
**When:** After core pipeline, before export  
**How:** `python main.py` runs this

### Component 3: Polish
**What:** Optimization + Export (Excel, PPT, Goal-based)  
**Why:** Actionable outputs + advanced features  
**When:** Last step  
**How:** Automatic with `main.py`

---

## Common Tasks

### Task 1: Run Full Pipeline
```bash
python main.py
```
**Output:** `outputs/` folder with Excel + PPT + JSON state

### Task 2: Run Without LLM (Testing)
```bash
python main.py --no-llm
```
**Use when:** Azure API is down, testing logic

### Task 3: Use Custom Data Directory
```bash
python main.py --data-dir ./my_data --output-dir ./my_results
```
**Use when:** Data is in non-standard location

### Task 4: Skip Exports (Faster Testing)
```bash
python main.py --no-excel --no-ppt
```
**Use when:** Only interested in state.json

### Task 5: Analyze Results Programmatically
```python
import json
with open("outputs/pipeline_state.json") as f:
    state = json.load(f)
print(f"EDA segments: {state['eda_segments']}")
print(f"Best model: {state['best_model']}")
```

---

## Key Functions & Where to Find Them

| Task | Module | Function | Returns |
|------|--------|----------|---------|
| Classify columns | `1_core/column_classifier.py` | `run_column_classification(df)` | `{column: category}` |
| Run EDA | `1_core/eda_enhanced.py` | `run_segment_eda(monthly_df, channels)` | `{segment: {rfe, corr, overlap}}` |
| Extract curves | `1_core/response_curves.py` | `run_response_curve_extraction(X, y, model, channels)` | `{channel: curve_data}` |
| Rank models | `1_core/modeling_enhanced.py` | `rank_models(models, y_actual, y_pred_list)` | `[top_10_models]` |
| Run full pipeline | `2_agentic/orchestrator.py` | `Orchestrator.run_full_pipeline()` | `PipelineState` |
| Get LLM client | `2_agentic/llm_integration.py` | `get_llm_client()` | `AzureOpenAI` |
| Generate narrative | `2_agentic/llm_integration.py` | `generate_eda_narrative(client, prompt)` | `str` |
| Optimize budget | `3_polish/mix_optimizer.py` | `run_mix_optimization(monthly_df, channels, elasticities)` | `{scenarios}` |
| Export Excel | `3_polish/export_excel.py` | `export_to_excel(path, eda_results, models, curves, scenarios)` | `None` (file written) |
| Export PPT | `3_polish/export_ppt.py` | `generate_ppt_presentation(path, narratives)` | `None` (file written) |
| Parse goals | `3_polish/goal_optimizer.py` | `run_goal_based_optimization(goal_text, base_allocation, elasticities)` | `{constraints, bounds}` |

---

## Data Flow

```
Raw CSVs
  ↓
[1] Column Classification
  ↓
[2] EDA (national + segment)
  ↓
[3] Outlier Detection & Removal
  ↓
[4] Feature Engineering
  ↓
[5] Model Training & Ranking
  ↓
[6] Response Curve Extraction
  ↓
[7] GenAI Narratives
  ↓
[8] Mix Optimization (4 scenarios)
  ↓
[9] Export (Excel + PPT)
  ↓
Excel + PowerPoint + JSON State
```

---

## Configuration Files

### `.env` (Required)
```env
AZURE_ENDPOINT=https://zs-eu1-ail-agentics-openai-team10.openai.azure.com/
AZURE_OPENAI_KEY=your_key_here
```

### `requirements.txt` (Dependencies)
```
pandas numpy scipy scikit-learn statsmodels
openpyxl python-pptx python-dotenv openai anthropic langchain
```

---

## Output Interpretation

### `pipeline_state.json`
```json
{
  "completed_steps": ["load_data", "classify_columns", "eda", ...],
  "eda_segments": ["National", "Category_A", "Category_B"],
  "models_trained": 25,
  "best_model": "Ridge",
  "response_curves": 9,
  "optimization_scenarios": 4,
  "errors": []
}
```

### Excel Sheets
| Sheet | Contains | Use For |
|-------|----------|---------|
| EDA Summary | National + segment stats | Understand data patterns |
| Models | Top-10 ranked models + coefs | Choose best model |
| Response Curves | Channel elasticities | Understand channel ROI |
| Optimization | 4 scenarios, budget shifts | Decision-making |
| Narratives | GenAI explanations | Stakeholder communication |

### PowerPoint Slides
| Slide # | Title | Purpose |
|---------|-------|---------|
| 1 | Title Slide | Introduction |
| 2 | Executive Summary | High-level findings |
| 3-5 | EDA Findings | Data insights |
| 6 | Outlier Removal | Data cleaning rationale |
| 7 | Feature Engineering | Feature decisions |
| 8 | Model Performance | Model selection |
| 9 | Optimization | Budget scenarios |
| 10 | Recommendations | Action items |

---

## Troubleshooting Checklist

- [ ] `.env` file exists with `AZURE_OPENAI_KEY`
- [ ] `data/` folder exists with all CSV files
- [ ] `requirements.txt` packages are installed
- [ ] Python 3.8+
- [ ] Azure API key is valid and not expired
- [ ] Try `python main.py --no-llm` to isolate issues

---

## Example: Custom Goal Optimization

```python
from src.3_polish.goal_optimizer import run_goal_based_optimization

goals = "Increase Digital ROI by 2%, keep TV under $40M"
result = run_goal_based_optimization(
    goals,
    base_allocation={"TV": 50e6, "Digital": 30e6, "SEM": 20e6},
    current_metrics={"TV": 2.5, "Digital": 2.0, "SEM": 1.8},
    elasticities={"TV": 0.5, "Digital": 0.8, "SEM": 0.6},
)

for constraint in result["applied_constraints"]:
    print(f"✅ {constraint}")
```

---

## Example: Custom Model Type

```python
from src.1_core.modeling_enhanced import fit_glm_poisson, rank_models

# Fit Poisson model
poisson = fit_glm_poisson(X, y)

# Get top-10 models
ranked = rank_models(
    [poisson, glm_gamma, ridge_model],
    y_actual,
    [pred_poisson, pred_gamma, pred_ridge]
)

print(f"Top model: {ranked[0]['model_id']} (score={ranked[0]['composite_score']})")
```

---

## Performance Notes

- **EDA:** < 1 min (1 year of monthly data)
- **Modeling:** 1-3 min (5-10 model types, 100 CV folds)
- **Optimization:** < 1 min (4 scenarios)
- **LLM Narratives:** 5-10 min (5 step narratives, ~5k tokens each)
- **Export:** < 1 min (Excel + PPT)
- **Total:** ~10-20 min (full pipeline with LLM)

---

## Useful Pandas Snippets

```python
# Load & explore data
import pandas as pd
df = pd.read_csv("data/Secondfile.csv")
print(df.describe())
print(df.corr()['total_gmv'].sort_values(ascending=False))

# Check for segments
print(df['Product_Category'].value_counts())

# Filter by date range
df_filtered = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2016-12-31')]
```

---

## Development Notes

- All modules use type hints for IDE support
- Docstrings follow Google style (use `help()` in Python)
- No external dependencies beyond those in `requirements.txt`
- Error handling is explicit (check `errors` in state)
- Logging is automatic (check logs in outputs folder)

---

**Last Updated:** Feb 16, 2026  
**Version:** 1.0  
**Status:** Production Ready ✅
