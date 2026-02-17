"""
=============================================================================
outlier_detection.py -- Data Cleaning, Validation & Outlier Detection
=============================================================================
Modules:
  1. Assumptions Registry
  2. Transaction-Level Cleaning
  3. Monthly Data Cleaning
  4. Weekly Data Cleaning
  5. Statistical Outlier Detection (IQR + Z-score)
  6. Business Context Review
  7. Reconciliation Checks
  8. Visualization
  9. Master Pipeline
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from config import (
    MEDIA_CHANNELS, MODEL_SETTINGS, get_paths, get_channel_cols, find_col, logger
)

plt.rcParams["figure.figsize"] = (14, 6)
sns.set_style("whitegrid")


# =============================================================================
# 1. ASSUMPTIONS REGISTRY
# =============================================================================

ASSUMPTIONS = {
    "A1_AUG2015": {
        "description": (
            "Aug 2015 shows extremely low GMV (<5% of median). "
            "Platform likely launched Jul 2015; Aug is ramp-up period."
        ),
        "decision": "EXCLUDE from modeling (insufficient data collection period)",
        "reasoning": (
            "The platform was new. Including a partial ramp-up month would "
            "distort elasticity estimates. Losing 1 of 12 months is costly, "
            "but keeping an anomalous point is worse for model validity."
        ),
    },
    "A2_SALE_DISCOUNTS": {
        "description": (
            "Discount spikes (30-40% of MRP) occur during known sale events "
            "(Diwali, Eid, Independence Day, etc.)."
        ),
        "decision": "KEEP all data. Create sale_intensity feature instead of removing.",
        "reasoning": (
            "Discounting during sales is standard e-commerce behavior, not a "
            "data quality issue. Removing these would eliminate the very signal "
            "we need to measure promotional elasticity."
        ),
    },
    "A3_CHANNEL_ZEROS": {
        "description": "Radio, Other, Content.Marketing show zeros in several months.",
        "decision": "KEEP zeros as legitimate 'no spend' periods. Use log(x+1) transform.",
        "reasoning": (
            "Not all channels are active every month. Zeros represent real "
            "budget decisions, not missing data. Log(x+1) handles these "
            "cleanly in regression without artificial imputation."
        ),
    },
    "A4_NPS_VARIATION": {
        "description": "NPS ranges 44-60 across 12 months.",
        "decision": "KEEP raw NPS values. Not outliers.",
        "reasoning": (
            "This range is normal for a growing e-commerce platform. "
            "NPS shows strong negative correlation with GMV (-0.96), but "
            "this is confounded by seasonality -- high-demand months have "
            "lower NPS due to delivery pressure. NPS is a control variable, "
            "not a causal driver in this context."
        ),
    },
    "A5_SEASONAL_PEAKS": {
        "description": (
            "Oct 2015, Sep 2015 show high GMV. Coincides with festival season "
            "(Dussehra, Diwali) and known sale events."
        ),
        "decision": "KEEP as normal seasonality. Sale_flag captures the effect.",
        "reasoning": (
            "Festival-season peaks are the core business pattern we are "
            "modeling. Removing them would eliminate the signal entirely."
        ),
    },
    "A6_DISCOUNT_ENDOGENEITY": {
        "description": "Discount may be endogenous (company discounts more when sales are low).",
        "decision": (
            "Use discount as CONTROL variable, not as primary MMIX lever. "
            "Create sale_intensity ratio instead."
        ),
        "reasoning": (
            "If the company increases discounts in response to low sales, "
            "the discount coefficient will be biased. Using it as a control "
            "rather than a lever avoids misinterpreting reverse causality."
        ),
    },
    "A7_SAMPLE_SIZE": {
        "description": (
            "Only 12 months of data (Jul 2015 -- Jun 2016). "
            "MMIX typically needs 60+ months for robust elasticities."
        ),
        "decision": (
            "Proceed with simpler models (max 3-4 predictors). "
            "No seasonal dummies (would consume all degrees of freedom). "
            "Use weekly aggregation to increase n to ~48."
        ),
        "reasoning": (
            "With n=12 (or 11 after Aug exclusion), degrees of freedom are "
            "severely limited. Weekly aggregation provides ~48 data points, "
            "enabling individual channel modeling and proper cross-validation."
        ),
    },
    "A8_SALES_GMV_DTYPE": {
        "description": (
            "Sales.csv GMV column is dtype 'object' -- contains whitespace or "
            "non-numeric entries. Needs coercion to float."
        ),
        "decision": "Coerce to numeric; NaN rows logged and removed.",
        "reasoning": (
            "Object dtype in a numeric field indicates data quality issues "
            "at source. Coercing to numeric and dropping failures is the "
            "standard safe approach."
        ),
    },
    "A9_RADIO_OTHER_SPURIOUS": {
        "description": "Radio and Other show strong negative correlation (-0.95) with GMV.",
        "decision": "Exclude from individual channel modeling or merge into residual bucket.",
        "reasoning": (
            "These channels have data only in a few months. The months they "
            "are active happen to be lower-GMV months, creating a false "
            "negative correlation. This is not causal -- Radio does not hurt "
            "sales. The pattern is an artifact of sparse, non-overlapping "
            "activity periods."
        ),
    },
}


def print_assumptions():
    """Print all documented assumptions."""
    print("=" * 70)
    print("DOCUMENTED ASSUMPTIONS FOR MMIX PIPELINE")
    print("=" * 70)
    for key, a in ASSUMPTIONS.items():
        print(f"\n  [{key}]")
        print(f"   Description: {a['description']}")
        print(f"   Decision:    {a['decision']}")
        print(f"   Reasoning:   {a['reasoning']}")
    print("=" * 70)


# =============================================================================
# 2. TRANSACTION-LEVEL CLEANING
# =============================================================================

def clean_transactions(df, dataset_name="transactions"):
    """
    Clean transaction-level data (firstfile.csv or Sales.csv).

    Removes: negative GMV, negative units, negative discount, NaN GMV,
             GMV=0 with units>0 (recording error).
    Flags:   discount > MRP (impossible but kept for review).

    Returns:
        (cleaned_df, log_list)
    """
    log = []
    n_start = len(df)
    logger.info("Cleaning %s (%s rows)", dataset_name, f"{n_start:,}")

    gmv_col = find_col(df, ["gmv_new", "GMV", "gmv"])
    units_col = find_col(df, ["units", "Units_sold", "Units"])
    mrp_col = find_col(df, ["product_mrp", "MRP", "mrp"])
    discount_col = find_col(df, ["discount", "Discount"])

    # Coerce GMV to numeric
    if gmv_col:
        before = df[gmv_col].isna().sum()
        df[gmv_col] = pd.to_numeric(df[gmv_col], errors="coerce")
        coerced = df[gmv_col].isna().sum() - before
        if coerced > 0:
            log.append({"step": "gmv_coerce", "rows_affected": coerced,
                        "reason": f"Coerced {coerced} non-numeric GMV to NaN"})
            logger.info("  [INFO] Coerced %d non-numeric GMV to NaN", coerced)

    # Negative GMV
    if gmv_col:
        n = (df[gmv_col] < 0).sum()
        if n > 0:
            df = df[df[gmv_col] >= 0]
            log.append({"step": "negative_gmv", "rows_removed": n,
                        "reason": f"Removed {n} negative GMV rows"})
            logger.info("  [OK] Removed %d negative GMV rows", n)

    # GMV=0 with units>0
    if gmv_col and units_col:
        df[units_col] = pd.to_numeric(df[units_col], errors="coerce")
        n = ((df[gmv_col] == 0) & (df[units_col] > 0)).sum()
        if n > 0:
            df = df[~((df[gmv_col] == 0) & (df[units_col] > 0))]
            log.append({"step": "zero_gmv_units", "rows_removed": n,
                        "reason": f"Removed {n} rows: GMV=0 but units>0"})
            logger.info("  [OK] Removed %d rows: GMV=0 but units>0", n)

    # Negative units
    if units_col:
        n = (df[units_col] < 0).sum()
        if n > 0:
            df = df[df[units_col] >= 0]
            log.append({"step": "negative_units", "rows_removed": n,
                        "reason": f"Removed {n} negative unit rows"})
            logger.info("  [OK] Removed %d negative unit rows", n)

    # Flag impossible discounts (discount > MRP)
    if discount_col and mrp_col:
        df[discount_col] = pd.to_numeric(df[discount_col], errors="coerce")
        df[mrp_col] = pd.to_numeric(df[mrp_col], errors="coerce")
        n = ((df[discount_col] > df[mrp_col]) & df[mrp_col].notna()).sum()
        if n > 0:
            df["flag_impossible_discount"] = (
                (df[discount_col] > df[mrp_col]) & df[mrp_col].notna()
            ).astype(int)
            log.append({"step": "impossible_discount", "rows_flagged": n,
                        "reason": f"Flagged {n} rows: discount > MRP (kept)"})
            logger.warning("  [WARN] Flagged %d rows: discount > MRP", n)

    # Negative discounts
    if discount_col:
        n = (df[discount_col] < 0).sum()
        if n > 0:
            df = df[df[discount_col] >= 0]
            log.append({"step": "negative_discount", "rows_removed": n,
                        "reason": f"Removed {n} negative discount rows"})
            logger.info("  [OK] Removed %d negative discount rows", n)

    # Drop NaN GMV
    if gmv_col:
        n = df[gmv_col].isna().sum()
        if n > 0:
            df = df.dropna(subset=[gmv_col])
            log.append({"step": "nan_gmv", "rows_removed": n,
                        "reason": f"Removed {n} NaN GMV rows"})
            logger.info("  [OK] Removed %d NaN GMV rows", n)

    n_end = len(df)
    removed = n_start - n_end
    pct = (removed / n_start * 100) if n_start > 0 else 0
    logger.info("  [SUMMARY] %s: %s -> %s (%s removed, %.2f%%)",
                dataset_name, f"{n_start:,}", f"{n_end:,}", f"{removed:,}", pct)

    return df, log


# =============================================================================
# 3. MONTHLY DATA CLEANING
# =============================================================================

def clean_monthly(df):
    """
    Validate and clean monthly aggregated data (SecondFile.csv).

    Returns:
        (cleaned_df, log_list, validation_dict)
    """
    log = []
    validation = {}
    logger.info("Cleaning Monthly Data (%d rows)", len(df))

    if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Duplicate dates
    if "Date" in df.columns:
        dups = df["Date"].duplicated().sum()
        validation["duplicate_dates"] = dups
        if dups > 0:
            df = df.drop_duplicates(subset="Date", keep="first")
            log.append({"step": "dup_dates", "rows_removed": dups,
                        "reason": f"Removed {dups} duplicate dates"})
            logger.warning("  [WARN] Removed %d duplicate dates", dups)
        else:
            logger.info("  [OK] No duplicate dates")

    # Aug 2015 exclusion (Assumption A1)
    aug_excluded = False
    if "Date" in df.columns and "total_gmv" in df.columns:
        aug = df[(df["Date"].dt.year == 2015) & (df["Date"].dt.month == 8)]
        if len(aug) > 0:
            median_gmv = df["total_gmv"].median()
            aug_gmv = aug["total_gmv"].values[0]
            ratio = aug_gmv / median_gmv if median_gmv > 0 else 0
            if ratio < 0.05:
                df = df[~((df["Date"].dt.year == 2015) & (df["Date"].dt.month == 8))]
                msg = (f"EXCLUDED Aug 2015: GMV={aug_gmv / 1e7:.2f}Cr is "
                       f"{ratio * 100:.1f}% of median {median_gmv / 1e7:.1f}Cr")
                log.append({"step": "aug_2015", "rows_removed": 1, "reason": msg})
                logger.info("  [EXCLUDE] %s", msg)
                aug_excluded = True
            else:
                logger.info("  [INFO] Aug 2015 GMV ratio %.1f%% -- keeping", ratio * 100)
    validation["aug_2015_excluded"] = aug_excluded

    # Negative values
    for col in df.select_dtypes(include=[np.number]).columns:
        n = (df[col] < 0).sum()
        if n > 0:
            df.loc[df[col] < 0, col] = 0
            log.append({"step": f"neg_{col}", "rows_affected": n,
                        "reason": f"Set {n} negative values in {col} to 0"})
            logger.warning("  [WARN] %s: %d negatives set to 0", col, n)

    # Channel spend validation (Assumption A3)
    channels = get_channel_cols(df)
    for ch in channels:
        df[ch] = pd.to_numeric(df[ch], errors="coerce").fillna(0)
    zeros = {ch: (df[ch] == 0).sum() for ch in channels}
    has_zeros = {ch: z for ch, z in zeros.items() if z > 0}
    validation["channel_zeros"] = zeros
    if has_zeros:
        logger.info("  [INFO] Channel zero-months (Assumption A3 -- no spend):")
        for ch, z in has_zeros.items():
            logger.info("         %-25s -> %d months zero (kept)", ch, z)

    # NPS validation
    if "NPS" in df.columns:
        nps_min, nps_max = df["NPS"].min(), df["NPS"].max()
        validation["nps_range"] = (nps_min, nps_max)
        if nps_min < -100 or nps_max > 100:
            logger.warning("  [WARN] NPS out of range: [%.1f, %.1f]", nps_min, nps_max)
        else:
            logger.info("  [OK] NPS range: %.1f to %.1f", nps_min, nps_max)

    # Discount sanity
    if "total_Discount" in df.columns and "total_Mrp" in df.columns:
        invalid = (df["total_Discount"] > df["total_Mrp"]).sum()
        if invalid > 0:
            df.loc[df["total_Discount"] > df["total_Mrp"], "total_Discount"] = \
                df.loc[df["total_Discount"] > df["total_Mrp"], "total_Mrp"]
            logger.warning("  [WARN] %d months: discount > MRP (capped)", invalid)
        else:
            logger.info("  [OK] All discounts <= MRP")
        validation["discount_valid"] = invalid == 0

    logger.info("  [SUMMARY] Monthly: %d rows remaining", len(df))
    if "Date" in df.columns and len(df) > 0:
        logger.info("            Range: %s to %s",
                     df["Date"].min().strftime("%b-%Y"),
                     df["Date"].max().strftime("%b-%Y"))

    return df, log, validation


# =============================================================================
# 4. WEEKLY DATA CLEANING
# =============================================================================

def clean_weekly(weekly_df):
    """
    Validate and clean weekly aggregated data.
    Called after data_aggregation builds the weekly dataset.

    Checks: duplicate weeks, partial weeks, zero-GMV weeks,
            anomalously low weeks, negative values, channel/NPS validation.

    Returns:
        (cleaned_df, log_list)
    """
    log = []
    n_start = len(weekly_df)
    df = weekly_df.copy()
    logger.info("Cleaning Weekly Data (%d weeks)", n_start)

    if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Duplicate weeks
    if "Date" in df.columns:
        dups = df["Date"].duplicated().sum()
        if dups > 0:
            df = df.drop_duplicates(subset="Date", keep="first")
            log.append({"step": "dup_weeks", "rows_removed": dups,
                        "reason": f"Removed {dups} duplicate weeks"})
            logger.warning("  [WARN] Removed %d duplicate weeks", dups)
        else:
            logger.info("  [OK] No duplicate weeks")

    # Partial weeks at boundaries (< 30% of median transactions)
    if "total_gmv" in df.columns and "n_transactions" in df.columns and len(df) > 4:
        median_txn = df["n_transactions"].median()
        threshold = median_txn * 0.3

        # First week
        if df.iloc[0]["n_transactions"] < threshold:
            label = df.iloc[0]["Date"].strftime("%Y-%m-%d")
            msg = (f"Removed first week ({label}): {df.iloc[0]['n_transactions']:.0f} "
                   f"txns vs median {median_txn:.0f} (partial)")
            df = df.iloc[1:].reset_index(drop=True)
            log.append({"step": "partial_start", "rows_removed": 1, "reason": msg})
            logger.info("  [EXCLUDE] %s", msg)

        # Last week
        if len(df) > 0 and df.iloc[-1]["n_transactions"] < threshold:
            label = df.iloc[-1]["Date"].strftime("%Y-%m-%d")
            msg = (f"Removed last week ({label}): {df.iloc[-1]['n_transactions']:.0f} "
                   f"txns vs median {median_txn:.0f} (partial)")
            df = df.iloc[:-1].reset_index(drop=True)
            log.append({"step": "partial_end", "rows_removed": 1, "reason": msg})
            logger.info("  [EXCLUDE] %s", msg)

    # Zero-GMV weeks
    if "total_gmv" in df.columns:
        n = (df["total_gmv"] <= 0).sum()
        if n > 0:
            dates = df[df["total_gmv"] <= 0]["Date"].dt.strftime("%Y-%m-%d").tolist()
            df = df[df["total_gmv"] > 0].reset_index(drop=True)
            log.append({"step": "zero_gmv_weeks", "rows_removed": n,
                        "reason": f"Removed {n} zero-GMV weeks: {dates}"})
            logger.info("  [EXCLUDE] Removed %d zero-GMV weeks", n)

    # Anomalously low weeks (< 5% of median)
    if "total_gmv" in df.columns and len(df) > 5:
        median = df["total_gmv"].median()
        mask = df["total_gmv"] < median * 0.05
        n = mask.sum()
        if n > 0:
            dates = df[mask]["Date"].dt.strftime("%Y-%m-%d").tolist()
            df = df[~mask].reset_index(drop=True)
            log.append({"step": "anomalous_weeks", "rows_removed": n,
                        "reason": f"Removed {n} anomalous weeks (GMV < 5% median): {dates}"})
            logger.info("  [EXCLUDE] Removed %d anomalous weeks", n)

    # Negative values
    for col in df.select_dtypes(include=[np.number]).columns:
        n = (df[col] < 0).sum()
        if n > 0:
            df.loc[df[col] < 0, col] = 0
            log.append({"step": f"neg_{col}", "rows_affected": n,
                        "reason": f"Set {n} negatives in {col} to 0"})
            logger.warning("  [WARN] %s: %d negatives set to 0", col, n)

    # Channel coercion
    for ch in get_channel_cols(df):
        df[ch] = pd.to_numeric(df[ch], errors="coerce").fillna(0)

    # NPS
    if "NPS" in df.columns:
        nps_min, nps_max = df["NPS"].min(), df["NPS"].max()
        if nps_min < -100 or nps_max > 100:
            logger.warning("  [WARN] NPS out of range: [%.1f, %.1f]", nps_min, nps_max)
        else:
            logger.info("  [OK] NPS range: %.1f to %.1f", nps_min, nps_max)

    n_end = len(df)
    logger.info("  [SUMMARY] Weekly: %d -> %d weeks (%d removed)",
                n_start, n_end, n_start - n_end)
    if "Date" in df.columns and len(df) > 0:
        logger.info("            Range: %s to %s",
                     df["Date"].min().strftime("%Y-%m-%d"),
                     df["Date"].max().strftime("%Y-%m-%d"))

    return df, log


# =============================================================================
# 5. STATISTICAL OUTLIER DETECTION
# =============================================================================

def detect_outliers_iqr(df, columns=None, multiplier=None):
    """
    Flag outliers using IQR method. Does NOT remove -- flags only.

    Returns:
        (outlier_flags_df, summary_dict)
    """
    multiplier = multiplier or MODEL_SETTINGS["iqr_multiplier"]
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info("IQR Outlier Detection (multiplier=%.1f)...", multiplier)
    flags = pd.DataFrame(index=df.index)
    summary = {}

    for col in columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all():
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        is_out = (vals < lo) | (vals > hi)
        flags[f"{col}_outlier"] = is_out.astype(int)
        n = is_out.sum()
        summary[col] = {"q1": q1, "q3": q3, "iqr": iqr,
                         "lower": lo, "upper": hi, "n_outliers": n}
        if n > 0:
            logger.info("  [FLAG] %s: %d outliers (bounds: [%.2f, %.2f]) values: %s",
                        col, n, lo, hi, vals[is_out].tolist())
        else:
            logger.info("  [OK]   %s: No outliers", col)

    return flags, summary


def detect_outliers_zscore(df, columns=None, threshold=None):
    """
    Flag outliers using Z-score. Does NOT remove -- flags only.

    Returns:
        (outlier_flags_df, summary_dict)
    """
    threshold = threshold or MODEL_SETTINGS["zscore_threshold"]
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info("Z-score Outlier Detection (threshold=%.1f)...", threshold)
    flags = pd.DataFrame(index=df.index)
    summary = {}

    for col in columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all():
            continue
        mean, std = vals.mean(), vals.std()
        if std == 0:
            logger.info("  [SKIP] %s: zero variance", col)
            continue
        z = np.abs((vals - mean) / std)
        is_out = z > threshold
        flags[f"{col}_zscore"] = is_out.astype(int)
        n = is_out.sum()
        summary[col] = {"mean": mean, "std": std, "n_outliers": n}
        if n > 0:
            logger.info("  [FLAG] %s: %d outliers (|z| > %.1f)", col, n, threshold)
        else:
            logger.info("  [OK]   %s: No outliers", col)

    return flags, summary


# =============================================================================
# 6. BUSINESS CONTEXT REVIEW
# =============================================================================

def business_context_review(df, special_sales_df=None):
    """
    Review each month/week against business context.
    Decisions: KEEP, EXCLUDE, or FLAG_REVIEW.

    Returns:
        list of decision dicts
    """
    logger.info("Business Context Review...")
    decisions = []

    if "Date" not in df.columns or "total_gmv" not in df.columns:
        logger.warning("  Date or total_gmv missing -- skipping")
        return decisions

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    median_gmv = df["total_gmv"].median()

    # Build sale period set
    sale_periods = set()
    if special_sales_df is not None:
        ss = special_sales_df.copy()
        if "Date" in ss.columns:
            if not pd.api.types.is_datetime64_any_dtype(ss["Date"]):
                ss["Date"] = pd.to_datetime(ss["Date"])
            sale_periods = set(ss["Date"].dt.to_period("M"))

    for _, row in df.iterrows():
        label = row["Date"].strftime("%b-%Y")
        gmv = row["total_gmv"]
        ratio = gmv / median_gmv if median_gmv > 0 else 0
        is_sale = row["Date"].to_period("M") in sale_periods

        if ratio < 0.10:
            d = {"month": label, "ratio": ratio, "is_sale": is_sale,
                 "action": "EXCLUDE",
                 "reason": f"GMV {ratio * 100:.1f}% of median -- data issue or ramp-up"}
        elif ratio > 2.0 and is_sale:
            d = {"month": label, "ratio": ratio, "is_sale": True,
                 "action": "KEEP",
                 "reason": f"High GMV ({ratio * 100:.0f}%) explained by sale events"}
        elif ratio > 2.0:
            d = {"month": label, "ratio": ratio, "is_sale": False,
                 "action": "FLAG_REVIEW",
                 "reason": f"High GMV ({ratio * 100:.0f}%) without known sale"}
        else:
            continue  # Normal -- skip

        decisions.append(d)
        tag = {"EXCLUDE": "[EXCLUDE]", "KEEP": "[KEEP]", "FLAG_REVIEW": "[FLAG]"}
        logger.info("  %s %s: %s", tag.get(d["action"], "[?]"), label, d["reason"])

    if not decisions:
        logger.info("  [OK] All periods within normal range")

    return decisions


# =============================================================================
# 7. RECONCILIATION CHECKS
# =============================================================================

def reconciliation_checks(monthly_df, investment_df=None, tolerance=0.05):
    """Validate roll-ups and cross-dataset consistency."""
    logger.info("Reconciliation Checks...")
    results = {}
    m = monthly_df.copy()

    # Revenue roll-up
    rev_cols = [c for c in m.columns if c.startswith("Revenue_")]
    if rev_cols and "total_gmv" in m.columns:
        diff = abs(m[rev_cols].sum(axis=1) - m["total_gmv"]) / m["total_gmv"] * 100
        mx = diff.max()
        results["revenue_rollup"] = mx
        if mx > tolerance * 100:
            logger.warning("  [WARN] Revenue roll-up max diff: %.2f%%", mx)
        else:
            logger.info("  [OK] Revenue roll-up valid (max diff: %.2f%%)", mx)

    # Units roll-up
    unit_cols = [c for c in m.columns if c.startswith("Units_") and c != "total_Units"]
    if unit_cols and "total_Units" in m.columns:
        diff = abs(m[unit_cols].sum(axis=1) - m["total_Units"]) / m["total_Units"] * 100
        mx = diff.max()
        results["units_rollup"] = mx
        if mx > tolerance * 100:
            logger.warning("  [WARN] Units roll-up max diff: %.2f%%", mx)
        else:
            logger.info("  [OK] Units roll-up valid (max diff: %.2f%%)", mx)

    # Investment roll-up
    channels = get_channel_cols(m)
    if channels and "Total.Investment" in m.columns:
        ch_sum = m[channels].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        total = pd.to_numeric(m["Total.Investment"], errors="coerce")
        diff = abs(ch_sum - total) / total * 100
        mx = diff.max()
        results["investment_rollup"] = mx
        if mx > tolerance * 100:
            logger.warning("  [WARN] Investment roll-up max diff: %.2f%%", mx)
        else:
            logger.info("  [OK] Investment roll-up valid (max diff: %.2f%%)", mx)

    # Cross-validate with MediaInvestment.csv
    if investment_df is not None:
        inv_col = find_col(investment_df, ["Total Investment", "Total.Investment"])
        if inv_col and "Total.Investment" in m.columns:
            logger.info("  [INFO] Cross-validating with MediaInvestment.csv...")
            inv_sum = pd.to_numeric(investment_df[inv_col], errors="coerce").sum()
            m_sum = pd.to_numeric(m["Total.Investment"], errors="coerce").sum()
            if m_sum > 0:
                cross = abs(inv_sum - m_sum) / m_sum * 100
                results["cross_dataset"] = cross
                if cross > tolerance * 100:
                    logger.warning("  [WARN] Cross-dataset diff: %.1f%%", cross)
                else:
                    logger.info("  [OK] Cross-dataset valid (%.1f%% diff)", cross)

    return results


# =============================================================================
# 8. VISUALIZATION
# =============================================================================

def plot_outlier_summary(df, outlier_flags, key_cols=None, save_dir=None):
    """Scatter plot with IQR bounds for key metrics."""
    save_dir = save_dir or get_paths()["plots_dir"]
    if key_cols is None:
        key_cols = ["total_gmv", "Total.Investment", "NPS"]
    key_cols = [c for c in key_cols if c in df.columns]
    if not key_cols:
        return

    n = len(key_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Outlier Detection Summary", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, key_cols):
        vals = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        flag_col = f"{col}_outlier"
        colors = (["red" if f else "#2196F3" for f in outlier_flags[flag_col]]
                  if flag_col in outlier_flags.columns
                  else ["#2196F3"] * len(vals))

        ax.scatter(range(len(vals)), vals, c=colors, s=80, edgecolors="white", zorder=3)
        ax.axhline(y=hi, color="red", ls="--", alpha=0.5, label=f"Upper: {hi:.0f}")
        ax.axhline(y=lo, color="red", ls="--", alpha=0.5, label=f"Lower: {lo:.0f}")
        ax.axhline(y=vals.median(), color="green", ls="-", alpha=0.5, label="Median")
        ax.set_title(col)
        ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "outlier_summary.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("  [OK] Outlier plot saved")


# =============================================================================
# 9. MASTER PIPELINE
# =============================================================================

def run_outlier_pipeline(data, granularity="monthly", save_dir=None):
    """
    Run the complete outlier detection and cleaning pipeline.

    Args:
        data:        dict of DataFrames from load_all_data()
        granularity: 'weekly' or 'monthly' (affects which validations run)
        save_dir:    plot output directory

    Returns:
        (clean_data_dict, log_list, assumptions_dict)
    """
    save_dir = save_dir or get_paths()["plots_dir"]

    print("=" * 70)
    print("[START] OUTLIER DETECTION & REMOVAL PIPELINE")
    print("=" * 70)

    full_log = []
    clean_data = {}

    print_assumptions()

    # --- Transactions ---
    if "transactions" in data:
        try:
            df, log = clean_transactions(data["transactions"].copy(), "transactions (firstfile.csv)")
            clean_data["transactions"] = df
            full_log.extend(log)
        except Exception as e:
            logger.error("Transaction cleaning failed: %s", e)
            clean_data["transactions"] = data["transactions"]

    if "sales" in data:
        try:
            df, log = clean_transactions(data["sales"].copy(), "sales (Sales.csv)")
            clean_data["sales"] = df
            full_log.extend(log)
        except Exception as e:
            logger.error("Sales cleaning failed: %s", e)
            clean_data["sales"] = data["sales"]

    # --- Monthly ---
    if "monthly" in data:
        try:
            df, log, validation = clean_monthly(data["monthly"].copy())
            clean_data["monthly"] = df
            full_log.extend(log)
        except Exception as e:
            logger.error("Monthly cleaning failed: %s", e)
            clean_data["monthly"] = data["monthly"]

        # Statistical outlier detection on monthly
        try:
            key_cols = ["total_gmv", "total_Units", "total_Discount", "NPS"]
            key_cols = [c for c in key_cols if c in clean_data["monthly"].columns]
            channels = get_channel_cols(clean_data["monthly"])
            detect_cols = key_cols + channels

            iqr_flags, _ = detect_outliers_iqr(clean_data["monthly"], detect_cols)
            detect_outliers_zscore(clean_data["monthly"], detect_cols)

            # Business context
            ss = data.get("special_sales")
            business_context_review(clean_data["monthly"], ss)

            # Reconciliation
            inv = data.get("investment")
            reconciliation_checks(clean_data["monthly"], inv)

            # Plot
            plot_outlier_summary(clean_data["monthly"], iqr_flags, save_dir=save_dir)
        except Exception as e:
            logger.error("Monthly outlier detection failed: %s", e)

    # --- Pass-through datasets ---
    for key in ["special_sales", "nps", "products", "investment"]:
        if key in data and key not in clean_data:
            clean_data[key] = data[key]

    # --- Summary ---
    print("\n" + "=" * 70)
    print("[DONE] OUTLIER PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Cleaning actions: {len(full_log)}")
    print(f"  Datasets: {list(clean_data.keys())}")
    if full_log:
        print("\n  Log:")
        for entry in full_log:
            removed = entry.get("rows_removed", entry.get("rows_affected",
                     entry.get("rows_flagged", 0)))
            print(f"    - [{entry['step']}] {entry['reason']} ({removed} rows)")

    return clean_data, full_log, ASSUMPTIONS


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    from eda_pipeline import load_all_data
    data = load_all_data()
    clean_data, log, assumptions = run_outlier_pipeline(data)
