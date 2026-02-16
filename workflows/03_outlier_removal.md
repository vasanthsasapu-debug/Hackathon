# Workflow: Outlier Detection & Removal

## TLDR
Identify statistical outliers per channel (IQR, zscore, or domain rules) → Remove or flag → Generate narrative rationale → Return cleaned DataFrame.

---

## Summary

### Objective
Remove extreme values that distort modeling without losing signal.

### Inputs
- DataFrame (from EDA workflow)
- Column classification dict
- Outlier config: method (IQR|zscore|domain), threshold, per_segment (bool)

### Engine Module
`src/mmix/engine/validation.py` (validation_remove_outliers) + future `outlier_detection.py`

### Steps
1. **Method Selection** → IQR (Q1-1.5×IQR, Q3+1.5×IQR) or zscore (|z| > 3)
2. **Apply per Channel** → For each Promotional_Activity column
3. **Apply per Segment** (optional) → If per_segment=True, outliers computed per segment
4. **Mark/Remove** → Flag rows or drop entirely
5. **Log Rationale** → Document how many rows, % of data, which channels most affected

### Outputs
```json
{
  "cleaned_df": DataFrame,
  "outliers_removed": {
    "total_rows": 150,
    "pct_of_data": 2.3,
    "by_channel": {
      "TV": 45,
      "Digital": 80,
      ...
    }
  },
  "method": "IQR",
  "per_segment": true
}
```

### Validation Rules
- ✅ Outlier % < 10% of data (else flag warning)
- ✅ >= 30 observations remain per segment
- ❌ Fail if >= 50% of data is outliers (likely data issue, not outliers)
- ✅ Sales_Output not heavily affected (warn if > 5% removed)

### Edge Cases
- **All values in a channel are same** → No outliers (skip)
- **Single segment with outliers** → Can be removed at segment level but not national
- **Time series seasonality** → IQR may over-flag; consider seasonal decomposition (V2)
- **Domain knowledge override** → Allow manual threshold specification

### Performance
- Outlier detection: <1 second
- Removal: Deterministic, reproducible

### Future Vision (V2)
- Integrate seasonal decomposition (STL)
- Domain-specific rules (e.g., "TV spend > $1M never an outlier")
- LLM critique: "Removing these outliers could hide X campaign impact"
