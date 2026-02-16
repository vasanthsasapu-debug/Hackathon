# Workflow: Exploratory Data Analysis (EDA)

## TLDR
Compute reach/frequency/engagement metrics per channel, segment, and national level → Analyze channel overlaps and sales correlations → Generate summaries for business interpretation.

---

## Summary

### Objective
Understand channel performance, customer overlap, and sales drivers before modeling.

### Inputs
- Cleaned DataFrame (from data_loading workflow)
- Column classification dict
- Segment list (from classification or user-defined)

### Engine Module
`src/mmix/engine/eda_metrics.py`

### Steps
1. **Segment Extraction** → Identify unique values in Demographic_Segment columns
2. **Reach Metrics** → Unique customers/entities per channel
3. **Frequency Metrics** → Avg interactions per customer per channel
4. **Engagement Metrics** → Spend/volume per customer per channel
5. **Channel Overlap** → % of customers reached by 2+ channels (e.g., TV AND Digital)
6. **Personal vs Digital Overlap** → Specific cross-channel analysis
7. **Sales Correlation** → Pearson/Spearman correlation between each channel and Sales_Output

### Outputs
```json
{
  "national": {
    "reach": {"TV": 10000, "Digital": 8000, ...},
    "frequency": {"TV": 2.5, "Digital": 3.2, ...},
    "engagement": {"TV": 150.5, "Digital": 200.3, ...},
    "overlap": {"TV_Digital": 0.35, ...},
    "sales_correlation": {"TV": 0.65, "Digital": 0.58, ...}
  },
  "segment_1": {...},
  "segment_2": {...}
}
```

### Validation Rules
- ✅ All numeric channel columns have >= 0 values
- ✅ Correlations in [-1, 1]
- ✅ Reach <= total customers
- ❌ Fail if < 30 observations per segment

### Edge Cases
- **No segments detected**: Treat entire dataset as "national"
- **Single customer in segment**: Skip that segment, flag in logs
- **Constant channel (all zeros)**: Correlation = NaN, flag as "no variance"
- **Missing channel data**: Treat as zero activity

### Performance
- EDA on 50MB data: <5 seconds
- Segment analysis: Linear in # of segments
