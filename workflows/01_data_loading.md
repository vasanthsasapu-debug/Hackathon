# Workflow: Data Loading & Column Classification

## TLDR
Load CSV file → Parse into DataFrame → Auto-classify columns into semantic categories (Time_Stamp, Entity_ID, Sales_Output, Promotional_Activity, Brand_Health, Demographic_Segment) → Validate schema → Return state.

---

## Summary

### Objective
Load raw transaction/aggregated data and semantically classify columns to enable downstream operations.

### Inputs
- CSV file path (e.g., `/data/Secondfile.csv`)
- Column classification rules (in `src/mmix/config.py`)

### Engine Module
`src/mmix/engine/column_classification.py`

### Steps
1. **Load CSV** → Pandas DataFrame
2. **Parse dtypes** → Infer numeric/categorical/datetime
3. **Classify columns** → Match against patterns (e.g., "month" → Time_Stamp, "sales|revenue|gmv" → Sales_Output)
4. **Validate** → Ensure required categories present (Time_Stamp, Entity_ID, Sales_Output must exist)
5. **Return** → Classified columns dict

### Outputs
```json
{
  "column_name": "semantic_category",
  "month": "Time_Stamp",
  "gmv_new": "Sales_Output",
  "product_category": "Demographic_Segment",
  "TV": "Promotional_Activity",
  ...
}
```

### Validation Rules
- ✅ CSV is readable (no corruption)
- ✅ At least one Time_Stamp column
- ✅ At least one Sales_Output column
- ❌ Fail if no columns match required categories

### Edge Cases
- **Missing Time_Stamp**: Raise ValueError
- **Multiple sales columns**: Use largest numeric column as primary Sales_Output
- **Unknown column type**: Classify as "Demographic_Segment" (safest default)
- **Date formats mixed**: Parse all as datetime, fail on unparseable

### Performance
- Load 50MB CSV: <2 seconds
- Classification: <100ms
