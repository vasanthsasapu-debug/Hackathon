# MMIX Data Flow Architecture

## TLDR
End-to-end pipeline: Load CSV → Classify columns in `/engine/column_classification.py` → Run EDA/Outlier/Features/Models/Curves/Optimization in `/engine/` modules → Orchestrate via `/agents/orchestrator.py` → Generate narratives via `/agents/llm.py` → Export to `/outputs/`.

---

## Full Data Flow

```
INPUT LAYER
├── /data/Secondfile.csv (or Sales.csv, firstfile.csv)
│   └── Load via orchestrator.node_load_data()

DETERMINISTIC ENGINE LAYER (src/mmix/engine/)
├── column_classification.py
│   └── Auto-classify columns: Time_Stamp, Entity_ID, Sales_Output, Promotional_Activity, Brand_Health, Demographic_Segment
│
├── eda_metrics.py
│   ├── Segment extraction
│   ├── Reach/Frequency/Engagement metrics
│   ├── Channel overlap analysis
│   └── Correlation analysis (sales vs channels)
│
├── modeling.py (consolidated)
│   ├── Feature selection (incremental)
│   ├── GLM (Poisson, Gamma)
│   ├── Bayesian, Fixed/Random Effects
│   ├── Cross-validation
│   └── model_scoring.py: Rank models (60% fit, 20% stability, 20% simplicity)
│
├── response_curves.py
│   ├── Curve fitting (linear, log, power-law, diminishing-return)
│   ├── Elasticity calculation
│   └── Confidence bands
│
├── optimization_engine.py
│   ├── Base Case (historical execution)
│   ├── Budget Neutral (within-channel optimization)
│   ├── Max Profit (cross-channel reallocation)
│   └── Blue Sky (unconstrained)
│
├── goal_parsing.py
│   └── NLP-based constraint extraction ("increase Email mROI by 2%")
│
├── validation.py
│   ├── Schema validation (required columns, dtypes)
│   ├── Ordinality checks (channels ordered correctly)
│   └── Metric consistency (no negative revenue, etc.)
│
└── utilities.py
    ├── Channel lookups
    └── Aggregation helpers

AGENTIC COORDINATION LAYER (src/mmix/agents/)
├── orchestrator.py
│   ├── State machine: PipelineState tracks end-to-end progress
│   ├── Node functions: call engine modules in sequence
│   │   ├── node_load_data()
│   │   ├── node_classify_columns() → engine/column_classification.py
│   │   ├── node_eda() → engine/eda_metrics.py
│   │   ├── node_outlier_removal() → engine/outlier_detection.py (legacy) or validation.py
│   │   ├── node_feature_engineering() → engine/feature_engineering.py (or advanced_modeling.py)
│   │   ├── node_modeling() → engine/modeling.py
│   │   ├── node_response_curves() → engine/response_curves.py
│   │   └── node_optimization() → engine/optimization_engine.py
│   │
│   └── Modes:
│       ├── deterministic: Run all engine modules, skip LLM narratives
│       └── agentic: Run all engine modules + LLM narratives + suggestions for loops
│
└── llm.py
    ├── Azure OpenAI client initialization
    └── Narrative generators:
        ├── generate_eda_narrative()
        ├── generate_outlier_narrative()
        ├── generate_feature_narrative()
        ├── generate_model_narrative()
        └── generate_optimization_narrative()

OUTPUT LAYER (outputs/, outtest/)
├── outputs/ (production)
│   ├── top_10_models.csv
│   ├── response_curves.xlsx (multi-sheet)
│   ├── mmix_report.pptx
│   └── optimization_scenarios.xlsx
│
└── outtest/ (experiments, intermediate validation)
    └── pipeline_state.json (state checkpoint)

CONFIG LAYER (src/mmix/config.py)
├── Centralized: channels, thresholds, weights, model families
├── Azure OpenAI: API keys, endpoints, deployment names
└── Pipeline: hyperparameters (VIF, R² thresholds, CV folds)
```

---

## Execution Flow (Simplified)

1. **main.py** (Entry point)
   - Parse args: `--data secondfile.csv --mode agentic --output-dir outputs/`
   - Load data

2. **orchestrator.py** (Coordination)
   - Initialize PipelineState
   - For each step in workflow:
     - Call engine module (deterministic)
     - Validate output (fail-fast)
     - If agentic mode: generate narrative
     - Log and checkpoint state

3. **engine/** (Computation)
   - All return deterministic outputs (DataFrames, dicts, plots)
   - No side effects unless intentional (writes to `/outputs/`)

4. **llm.py** (Narratives, if agentic)
   - Called only if `mode == "agentic"`
   - Receives pre-computed engine outputs
   - Generates context-rich summaries

5. **export/** (Delivery)
   - Called at end of pipeline
   - Writes Excel, PPT, CSVs to `/outputs/`
   - Orchestrator decides what to export

---

## Key Principles

- **Separation of Concerns**: Engine = computation, Agents = coordination, LLM = narrative
- **Determinism**: No randomness in engine modules (fixed seeds, explicit CV folds)
- **Validation**: Fail-fast on schema/ordinality/metric errors
- **Reproducibility**: Same input → same output, always
- **Testability**: Engine modules independently testable
- **Config Centralization**: Single source of truth for all parameters
